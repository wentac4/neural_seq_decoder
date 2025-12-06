
modelName = 'speech_time_masking'

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
args['whiteNoiseSD'] = 1.2 #0.8 (Shengguang optimal) (can try 1.0) 
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['maxMaskLength'] = 16 # new added (optimal)
args['nMasks'] = 2 # new added (optimal)
args['mask_noise_sd'] = 1 # new added

from neural_decoder.neural_decoder_trainer_time_masking import trainModel

trainModel(args)