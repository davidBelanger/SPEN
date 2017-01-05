
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
require 'SSVM'

package.path = package.path .. ';losses/?.lua'
require 'SquaredLossPerBatchItem'

local training_method = "E2E"

config.batch_size = 15
config.length = 10
config.domain_size = 2
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape

config.feature_width = 3
config.feature_hid_size = 11
config.energy_hid_size = 5
config.local_potentials_scale = 2.5
config.pairwise_potentials_scale = 1.5
config.initialize_uniformly = false


local data_config = {
	num_test_batches = 100,
    num_train_batches = 1000,
	preprocess_train = function (t) return {Inference1DUtil:make_onehots(t[1],config.y_shape),t[2]} end
}

local problem = ChainCRFSequenceTagging(config, data_config)

local spen_model = ChainSPEN(config)

local optimization_config =   {
    num_iterations =  30,
    learning_rate_scale = 0.5,
    momentum_gamma = 0.66,
    line_search = false,
    init_line_search_step = 1,
    rtol = 0.0001,
    learning_rate_decay = 0.50,
    return_optimization_diagnostics = true
}

local gd_inference_config = {
    return_objective_value = false,
    entropy_weight = 1.0,
    logit_iterates = false,
    mirror_descent = true,
    return_optimization_diagnostics = false,
    optimization_config = optimization_config
}
local gd_prediction_net = GradientBasedInference(y_shape, gd_inference_config):spen_inference(spen_model)
local spen_predictor = RoundingPredictor(gd_prediction_net,y_shape)

local loss_wrapper
if(training_method == "E2E") then
    loss_wrapper = Independent(gd_prediction_net,nn.BCECriterion(), nil, nil)
elseif(training_method == "SSVM") then
    la_gd_config = Util:copyTable(gd_inference_config)
    la_gd_config.return_objective_value = true
    local ndims = 3
    local cost_net = SquaredLossPerBatchItem(ndims)
    local loss_augmented_predictor = GradientBasedInference(y_shape, la_gd_config):loss_augmented_spen_inference(spen_model,cost_net)
    local full_energy_net = spen_model:full_energy_net()
    loss_wrapper = SSVM(full_energy_net,loss_augmented_predictor)
else
    assert(false,'invalid training method')
end

local test_batcher = problem:get_test_batcher()

local bayes_evaluator = HammingEvaluator(test_batcher,function(x) return problem.crf_model:predict(x) end)
local bayes_accuracy = bayes_evaluator:evaluate('Bayes')
local evaluator = HammingEvaluator(test_batcher, function(x) return spen_predictor:predict(x) end)

local evaluate = {
	epochHookFreq = 1,
	hook = function(i) return evaluator:evaluate(i) end
}


local evaluate = Callback(function(data) return evaluator:evaluate(data.epoch) end, 10)
local callbacks = {evaluate}

local optimization_config = {
    num_epochs = 25,
    batches_per_epoch = 500,
    opt_config = {learningRate=0.001},
    gradient_clip = 2.0,
    opt_state = {},   
    opt_method = optim.adam,
    callbacks = {evaluate},
    modules_to_update = gd_prediction_net
}

local training_config = {
    batches_per_epoch = 10,
    batch_size = config.batch_size,
    num_epochs = 50
}
local batcher = problem:get_train_batcher()

Train(loss_wrapper,batcher,optimization_config, training_config, callbacks):train()

local final_acc = evaluator:evaluate('final')

--assert that it can get within 5% of the Bayes error
assert((bayes_accuracy - final_acc)/bayes_accuracy < 0.05)



