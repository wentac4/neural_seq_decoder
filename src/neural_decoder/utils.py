import torch
from lhotse.dataset.signal_transforms import time_warp as time_warp_impl

import random
from typing import Optional

# From https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
def make_pad_mask(
    lengths: torch.Tensor,
    max_len: int = 0,
    pad_left: bool = False,
) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
      pad_left:
        If ``False`` (default), padding is on the right.
        If ``True``, padding is on the left.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expanded_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    if pad_left:
        mask = expanded_lengths < (max_len - lengths).unsqueeze(1)
    else:
        mask = expanded_lengths >= lengths.unsqueeze(-1)

    return mask

# From https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py
def time_warp(
    features: torch.Tensor,
    p: float = 0.9,
    time_warp_factor: Optional[int] = 80,
    supervision_segments: Optional[torch.Tensor] = None,
    seed=0
):
    """Apply time warping on a batch of features"""
    random.seed(seed)
    if time_warp_factor is None or time_warp_factor < 1:
        return features
    assert (
        len(features.shape) == 3
    ), f"SpecAugment only supports batches of single-channel feature matrices. {features.shape}"
    features = features.clone()
    if supervision_segments is None:
        # No supervisions - apply spec augment to full feature matrices.
        for sequence_idx in range(features.size(0)):
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            features[sequence_idx] = time_warp_impl(
                features[sequence_idx], factor=time_warp_factor
            )
    else:
        # Supervisions provided - we will apply time warping only on the supervised areas.
        for sequence_idx, start_frame, num_frames in supervision_segments:
            if random.random() > p:
                # Randomly choose whether this transform is applied
                continue
            end_frame = start_frame + num_frames
            features[sequence_idx, start_frame:end_frame] = time_warp_impl(
                features[sequence_idx, start_frame:end_frame], factor=time_warp_factor
            )

    return features
