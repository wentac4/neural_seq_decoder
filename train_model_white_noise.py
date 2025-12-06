
modelName = 'speech_white_noise'

args = {}
args['outputDir'] = '/content/drive/MyDrive/NSP/neural_seq_decoder-master/logs/speech_logs/' + modelName
args['datasetPath'] = '/content/drive/MyDrive/NSP/neural_seq_decoder-master/data/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.05
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 1.2
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['maxMaskLength'] = 0
args['nMasks'] = 0

from neural_decoder.neural_decoder_trainer_time_masking import trainModel

trainModel(args)