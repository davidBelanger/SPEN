
require 'Imports'
require 'optim'
package.path = package.path .. ';model/?.lua'
require 'ChainSPEN'

package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'

package.path = package.path .. ';infer/?.lua'
require 'GradientBasedInference'
require 'RoundingPredictor'

package.path = package.path .. ';problem/?.lua'
require 'ChainCRFSequenceTagging'

package.path = package.path .. ';train/?.lua'
require 'Train'
require 'Independent'

package.path = package.path .. ';losses/?.lua'
require 'SquaredLossPerBatchItem'

config.batch_size = 5
config.length = 10
config.domain_size = 2
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape

config.feature_width = 3
config.feature_hid_size = 5
config.energy_hid_size = 5 --this won't get used
config.local_potentials_scale = 5.5
config.pairwise_potentials_scale = 2.0


--todo: give this a lambda for preprocessing the examples
local data_config = {
    num_test_batches = 100,
    num_train_batches = 1500,
    preprocess_train = nil
}

local problem = ChainCRFSequenceTagging(config, data_config)

local spen_model = ChainSPEN(config)

local classifier_net = spen_model.classifier_network
local spen_predictor = RoundingPredictor(classifier_net,y_shape)

local training_net = nn.Sequential():add(classifier_net):add(nn.Log())

local criterion = nn.ClassNLLCriterion()
local preprocess_ground_truth = nn.Reshape(config.batch_size*config.length,false)
local preprocess_prediction = nn.Reshape(config.batch_size*config.length,config.domain_size,false)
local loss_wrapper = Independent(training_net,criterion, preprocess_ground_truth, preprocess_prediction)

local test_batcher = problem:get_test_batcher()

local bayes_evaluator = HammingEvaluator(test_batcher,function(x) return problem.crf_model:predict(x) end)
local bayes_accuracy = bayes_evaluator:evaluate('Bayes')

local evaluator = HammingEvaluator(test_batcher, function(x) return spen_predictor:predict(x) end)



local optimization_config = {
    num_epochs = 25,
    batches_per_epoch = 500,
    opt_config = {learningRate=0.001},
    gradient_clip = 2.0,
    opt_state = {},   
    opt_method = optim.adam,
    modules_to_update = training_net
}

local training_config = {
    batches_per_epoch = 100,
    batch_size = config.batch_size,
    num_epochs = 50
}
local batcher = problem:get_train_batcher()

Train(loss_wrapper,batcher,optimization_config, training_config):train()
local final_acc = evaluator:evaluate(0)
--assert that it can get within 5% of the Bayes error
assert((bayes_accuracy - final_acc)/bayes_accuracy < 0.05)



