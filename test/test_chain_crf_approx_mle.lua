
require 'Imports'
require 'optim'
package.path = package.path .. ';model/?.lua'
require 'ChainCRF'
package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'
require 'SumProductInference'
require 'MeanFieldInference'

package.path = package.path .. ';problem/?.lua'
require 'ChainCRFSequenceTagging'

package.path = package.path .. ';infer/?.lua'
require 'GradientBasedInference'
require 'RoundingPredictor'

package.path = package.path .. ';train/?.lua'
require 'Train'
require 'Independent'

inference_method = "GD"
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
	num_test_batches = 100,
    num_train_batches = 1000,
    preprocess_train = function (t) return {t[1]:double():reshape(config.batch_size*config.length),t[2]} end
}


local problem = ChainCRFSequenceTagging(config, data_config)
local crf_model = ChainCRF(config)


local node_marginals_net
if(inference_method == "MF") then
    local mf_config = {
        num_iters = 15
    }   
    node_marginals_net = nn.Sequential():add(crf_model.log_edge_potentials_network):add(MeanFieldInference(y_shape,mf_config):inference_net())
elseif(inference_method == "GD") then
    local unconstrained_optimization_config =   {
        num_iterations =  15,
        learning_rate_scale = 0.5,
        momentum_gamma = 0,
        line_search = true,
        init_line_search_step = 1,
        rtol = 0.001,
        learning_rate_decay = 0.50,
        return_optimization_diagnostics = true
    }
    local gd_config = {
        return_objective_value = false,
        entropy_weight = 1.0,
        logit_iterates = true,
        return_optimization_diagnostics = false,
        optimization_config = unconstrained_optimization_config
    }
    node_marginals_net = GradientBasedInference(y_shape, gd_config):spen_inference(crf_model)
else
    assert(false)
end
local spen_predictor = RoundingPredictor(node_marginals_net,y_shape)

local training_net = nn.Sequential():add(node_marginals_net):add(nn.Reshape(config.batch_size*config.length,config.domain_size,false)):add(nn.Log())
local criterion = nn.ClassNLLCriterion()
local loss_wrapper = Independent(training_net,criterion)

local test_batcher = problem:get_test_batcher()

local bayes_evaluator = HammingEvaluator(test_batcher,function(x) return problem.crf_model:predict(x) end)
local bayes_accuracy = bayes_evaluator:evaluate('Bayes')

local evaluator = HammingEvaluator(test_batcher, function(x) return crf_model:predict(x) end)


local evaluate = {
	epochHookFreq = 1,
	hook = function(i) return evaluator:evaluate(i) end
}

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

local final_acc = problem:evaluate(function(x) return crf_model:predict(x) end,'final')

--assert that it can get within 2% of the Bayes error
assert((bayes_accuracy - final_acc) < 0.02)



