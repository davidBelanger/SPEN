
require 'Imports'
require 'optim'
package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'

package.path = package.path .. ';model/?.lua'
require 'GraphSPEN'
require 'BinaryGraphSPEN'

package.path = package.path .. ';infer/?.lua'
require 'GradientBasedInference'
require 'RoundingPredictor'

package.path = package.path .. ';problem/?.lua'
require 'SimpleBinaryLinkPrediction'
require 'SimpleLabeledLinkPrediction'

package.path = package.path .. ';train/?.lua'
require 'End2End'
require 'Train'
require 'SSVM'

package.path = package.path .. ';losses/?.lua'
require 'SquaredLossPerBatchItem'

local training_method = "E2E"

local config = {}
config.batch_size = 5
config.num_nodes = 6
config.feature_size = 6
config.domain_size = 2  
local y_shape = {config.batch_size,config.num_nodes,config.num_nodes,config.domain_size}
config.y_shape = y_shape
config.= {}
config.feature_hid_size = 4
config.temperature = 1.0
config.initialize_uniformly = true

local joiner = nn.JoinTable(4)
local function make_onehots(t)
    local slices = {}
    for i = 1,config.domain_size do
        table.insert(slices,t:eq(i):double():view(config.batch_size,config.num_nodes,config.num_nodes,1))
    end
    return joiner:forward(slices):clone()
end
local data_config = {
	num_test_batches = 100,
}

data_config.preprocess_test = nil
if(config.domain_size == 2 or training_method == "SSVM") then
    data_config.preprocess_train = function (t) return {make_onehots(t[1]),t[2]} end
else
    data_config.preprocess_train = function (t) return {t[1]:view(config.batch_size*config.num_nodes*config.num_nodes),t[2]} end
end

if(config.domain_size == 2) then
    problem = SimpleLabeledLinkPrediction(config, data_config)
    spen_model = GraphSPEN(config)

    --spen_model = BinaryGraphSPEN(config)
else
    problem = SimpleLabeledLinkPrediction(config, data_config)
    spen_model = GraphSPEN(config)
end

local optimization_config =   {
    num_iterations =  15,
    learning_rate_scale = 0.5,
    momentum_gamma = 0,
    line_search = true,
    init_line_search_step = 1,
    rtol = 0.0001,
    learning_rate_decay = 0.50,
    return_optimization_diagnostics = true
}

local gd_inference_config = {
    return_objective_value = false,
    entropy_weight = 1.0,
    logit_iterates = true,
    mirror_descent = false,
    return_optimization_diagnostics = false,
    optimization_config = optimization_config
}
local gd_prediction_net = GradientBasedInference(y_shape, gd_inference_config):spen_inference(spen_model)
local spen_predictor = RoundingPredictor(gd_prediction_net,y_shape)

local loss_wrapper
if(training_method == "E2E") then
    local crit
    local training_net
    if(config.domain_size == 2) then
        crit = nn.BCECriterion()
        training_net = gd_prediction_net
    else
        crit = nn.ClassNLLCriterion()
        training_net = nn.Sequential():add(gd_prediction_net):add(nn.Reshape(config.batch_size*config.num_nodes*config.num_nodes,config.domain_size,false)):add(nn.Log())
    end
    loss_wrapper = End2End(training_net,crit)
elseif(training_method == "SSVM") then
    la_gd_config = Util:copyTable(gd_inference_config)
    la_gd_config.return_objective_value = true

    local ndims = 4
    local cost_net = SquaredLossPerBatchItem(ndims)
    local loss_augmented_predictor = GradientBasedInference(y_shape, la_gd_config):loss_augmented_spen_inference(spen_model,cost_net)
    local full_energy_net = spen_model:full_energy_net()
    loss_wrapper = SSVM(full_energy_net,loss_augmented_predictor)
else
    assert(false,'invalid training method')
end

--TODO: make SimpleLinkPrediction have a predict method
local bayes_accuracy = problem:evaluate(function(x) return problem:predict(x) end,'Bayes')

local evaluate = {
	epochHookFreq = 1,
	hook = function(i) return problem:evaluate(function(x) return spen_predictor:predict(x) end,i) end
}

local training_config = {
    numEpochs = 150,
    batchesPerEpoch = 50,
    batch_size = config.batch_size,
    optConfig = {learningRate=0.001},
    optState = {},   
    optMethod = optim.adam,
    gradientClip = 5.0,
    epochHooks = {evaluate}
}

local function batcher() return problem:get_train_batch() end
Train(loss_wrapper,batcher,training_config):train()
local final_acc = problem:evaluate(function(x) return spen_predictor:predict(x) end,'final')

--assert that it can get within 5% of the Bayes error
assert((bayes_accuracy - final_acc)/bayes_accuracy < 0.05)



