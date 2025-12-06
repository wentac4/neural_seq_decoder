import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F

from lhotse.dataset import SpecAugment

from .model import GRUDecoder
from .dataset import SpeechDataset

from .utils import make_pad_mask, time_warp
from .signed_log_transform_augmentations import SignedLog1pTransform


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    ############################################################
    feature_transform = SignedLog1pTransform()

    train_ds = SpeechDataset(loadedData["train"], transform=feature_transform)
    test_ds = SpeechDataset(loadedData["test"], transform=feature_transform)
    ############################################################

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    ################################################################################
    use_cr_ctc = False
    cr_loss_scale = 0.2
    use_spec_augment = False
    time_masking_factor = 1
    use_time_warp = False

    if "use_cr_ctc" in args:
        use_cr_ctc = args["use_cr_ctc"]
    if "cr_loss_scale" in args:
        cr_loss_scale = args["cr_loss_scale"]
    if "use_spec_augment" in args:
        use_spec_augment = args["use_spec_augment"]
    if "time_masking_factor" in args:
        time_masking_factor = args["time_masking_factor"]
    if "use_time_warp" in args:
        use_time_warp = args["use_time_warp"]

    if use_spec_augment:
        # SpecAugment for CR-CTC, used with 2x-repeated batch
        spec_augment = SpecAugment(
            time_warp_factor=0,                         # Do time warping separately, if at all
            num_frame_masks=10*time_masking_factor,             # default: 10
            features_mask_size=27,
            num_feature_masks=2,
            frames_mask_size=100,
            max_frames_mask_fraction=0.15*time_masking_factor,  # default: 0.15
        ).to(device)
    
    # Use "none" so we can normalize ourselves
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    ################################################################################

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=args["lrEnd"] / args["lrStart"],
        total_iters=args["nBatch"],
    )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        ################################################################################
        B = X.shape[0]
        X_base = X
        
        if use_time_warp:
            X_base = time_warp(
                X_base,
                time_warp_factor=80,
                supervision_segments=None,
                seed=args["seed"]
            )

        if args["whiteNoiseSD"] > 0:
            X_base = X_base + torch.randn(X_base.shape, device=device) * args["whiteNoiseSD"]

        if args["constantOffsetSD"] > 0:
            X_base = X_base + (
                torch.randn([X_base.shape[0], 1, X_base.shape[2]], device=device)
                * args["constantOffsetSD"]
            )

        ##################################################
        #  CR-CTC TRAINING
        ##################################################
        if use_cr_ctc:
            # Repeat batch 2×
            X_rep = X_base.repeat(2, 1, 1)
            X_len_rep = X_len.repeat(2)
            y_rep = torch.cat([y, y], dim=0)
            y_len_rep = torch.cat([y_len, y_len], dim=0)
            dayIdx_rep = dayIdx.repeat(2)

            # Apply SpecAugment to 2B batch
            if use_spec_augment:
                X_rep = spec_augment(X_rep)

            # Single forward pass on 2B batch
            pred_rep = model.forward(X_rep, dayIdx_rep)
            log_probs_rep = pred_rep.log_softmax(2)
            out_lens_rep = ((X_len_rep - model.kernelLen) / model.strideLen).to(torch.int32)

            # CTC loss
            ctc_per_seq_2B = loss_ctc(
                torch.permute(log_probs_rep, [1, 0, 2]),
                y_rep,
                out_lens_rep,
                y_len_rep,
            )

            # Reshape
            ctc_per_seq_2xB = ctc_per_seq_2B.view(2, B)
            y_len_2xB = y_len_rep.view(2, B).to(ctc_per_seq_2xB.dtype)

            # Convert to "mean": divide each per-seq loss by its own target length, then mean
            ctc_loss_mean = (ctc_per_seq_2xB / y_len_2xB).mean()

            # Symmetric KL for consistency regularization
            exchanged_targets = torch.roll(log_probs_rep.detach(), B, dims=0)

            cr_raw = F.kl_div(
                input=log_probs_rep,
                target=exchanged_targets,
                reduction="none",
                log_target=True,
            )

            # Mask invalid frames
            length_mask = make_pad_mask(out_lens_rep, max_len=cr_raw.size(1)).unsqueeze(-1)
            cr_raw = cr_raw.masked_fill(length_mask, 0.0)

            cr_sum = cr_raw.sum()
            num_valid = (~length_mask).sum()
            cr_loss_mean = cr_sum / num_valid
            cr_loss_mean = cr_loss_mean * 0.5

            # Final CR-CTC loss
            loss = ctc_loss_mean + cr_loss_scale * cr_loss_mean

        ##################################################
        #  VANILLA CTC TRAINING (if use_cr_ctc = False)
        ##################################################
        else:
            pred = model.forward(X_base, dayIdx)

            ctc_per_seq = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                y_len,
            )
            loss = (ctc_per_seq / y_len.to(ctc_per_seq.dtype)).mean()
        ################################################################################

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    ################################################################################
                    ctc_per_seq = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = (ctc_per_seq / y_len.to(ctc_per_seq.dtype)).mean()
                    ################################################################################
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()