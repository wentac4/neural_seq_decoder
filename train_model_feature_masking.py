
modelName = 'speech_feature_masking'

args = {}
args['outputDir'] = '/content/drive/MyDrive/NSP/neural_seq_decoder-master/logs/speech_logs/' + modelName
args['datasetPath'] = '/content/drive/MyDrive/NSP/neural_seq_decoder-master/data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128 #64
args['lrStart'] = 0.05 #0.02
args['lrEnd'] = 0.02
args['nUnits'] = 256 #1024
args['nBatch'] = 10000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2 #0.4
args['whiteNoiseSD'] = 1.2 #0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['maxMaskLength'] = 32 # new added
args['nMasks'] = 4 # new added 
args['mask_noise_sd'] = 1 # new added

from neural_decoder.neural_decoder_trainer_feature_masking import trainModel

trainModel(args)