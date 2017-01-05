
require 'Imports'
require 'optim'
package.path = package.path .. ';model/?.lua'
require 'ChainCRF'
package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'
require 'SumProductInference'

package.path = package.path .. ';problem/?.lua'
require 'ChainCRFSequenceTagging'

package.path = package.path .. ';train/?.lua'
require 'MLE'
require 'Train'

config.batch_size = 5
config.length = 10
config.domain_size = 5
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape

config.feature_width = 3
config.feature_hid_size = 5
config.local_potentials_scale = 2.5
config.pairwise_potentials_scale = 3.0

local data_config = {
    num_test_batches = 250,
    num_train_batches = 2500,
}

local problem = ChainCRFSequenceTagging(config, data_config)

local crf_model = ChainCRF(config)
local preprocess_net = crf_model.log_edge_potentials_network
local logZ_net = SumProductInference(config.y_shape):inference_net()
local score_net = crf_model.energy_from_potentials_network
local preprocess_labels_net = nn.Sequential():add(nn.Reshape(config.batch_size,config.length,false)):add(nn.OneHot(config.domain_size))
local loss_wrapper = MLE(preprocess_net, score_net, logZ_net, preprocess_labels_net)

local test_batcher = problem:get_test_batcher()

local bayes_evaluator = HammingEvaluator(test_batcher,function(x) return problem.crf_model:predict(x) end)
local bayes_accuracy = bayes_evaluator:evaluate('Bayes')

local evaluator = HammingEvaluator(test_batcher, function(x) return crf_model:predict(x) end)

local evaluate = Callback(function(i) return evaluator:evaluate(i) end,1)

local optimization_config = {
    num_epochs = 25,
    batches_per_epoch = 500,
    opt_config = {learningRate=0.001},
    gradient_clip = 2.0,
    opt_state = {},   
    opt_method = optim.adam,
    callbacks = {evaluate},
    modules_to_update = crf_model.log_edge_potentials_network
}

local training_config = {
    batches_per_epoch = 10,
    batch_size = config.batch_size,
    num_epochs = 50
}
local batcher = problem:get_train_batcher()
Train(loss_wrapper,batcher,optimization_config, training_config):train()
local final_acc = evaluator:evaluate('final')

--assert that it can get within 5% of the Bayes error
print(bayes_accuracy, final_acc)
assert((bayes_accuracy - final_acc) < 0.05)



