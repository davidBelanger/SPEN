local seed = 0
torch.manualSeed(seed)
require 'nn'
require 'nngraph'
require 'optim'

package.path = package.path .. ';util/?.lua'

require 'LogSumExp'
require 'Cond'
require 'EpochDropout'
require 'Predicate'
require 'PrintNoNewline'
require 'TruncatedBackprop'
require 'IO'
require 'Callback'
require 'OneHot'
require 'SquaredLossPerBatchItem'
require 'Saver'
require 'RepeatedCriterion'

package.path = package.path .. ';optimize/?.lua'

require 'GradientDirection'
require 'LineSearch'
require 'UnrolledGradientOptimizer'

package.path = package.path .. ';../torch-util/?.lua'
require 'Entropy'
require 'Util'
require 'Constant'
require 'Print'
require 'MinibatcherFromFileList'
require 'OnePassMiniBatcherFromFileList'

package.path = package.path .. ';evaluate/?.lua'

require 'Evaluator'
require 'HammingEvaluator'
require 'MultiLabelEvaluation'
require 'SRLEvaluator'
require 'PSNREvaluator'


package.path = package.path .. ';batch/?.lua'
require 'BatcherFromFactory'
require 'BatcherFromFile'
require 'SRLBatcher'

package.path = package.path .. ';train/?.lua'
require 'Train'
require 'Independent'
require 'SSVM'
require 'InstanceWeightedNLL'
require 'TrainingWrappers'

package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'

package.path = package.path .. ';model/?.lua'
require 'ChainSPEN'
require 'MLCSPEN'
require 'SRLSPEN'
require 'DepthSPEN'

package.path = package.path .. ';infer/?.lua'
require 'GradientBasedInference'
require 'RoundingPredictor'
require 'GradientBasedInferenceConfig'

package.path = package.path .. ';flags/?.lua'
require 'GeneralOptions'
