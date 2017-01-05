local seed = 0
torch.manualSeed(seed)

require 'nn'
require 'nngraph'
--require 'Imports'

package.path = package.path .. ';../torch-util/?.lua'
require 'Util'

package.path = package.path .. ';util/?.lua'
require 'ReshapeAs'
require 'Constant'
require 'LogSumExp'
require 'Entropy'
require 'TruncatedBackprop'



package.path = package.path .. ';optimize/?.lua'
require 'UnrolledGradientOptimizer'

package.path = package.path .. ';model/?.lua'
require 'ChainCRF'

package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'
require 'SumProductInference'
require 'ExactInference'
require 'MeanFieldInference'

package.path = package.path .. ';infer/?.lua'
require 'GradientBasedInference'

local config = {}
config.batch_size = 5
config.length = 4
config.domain_size = 2
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}

config.y_shape = y_shape
config.feature_width = 3
config.feature_hid_size = 5
config.local_potentials_scale = 4.0
config.pairwise_potentials_scale = 1.0
config.data_independent_transitions = false

local params = {init_at_local_prediction = false}
local crf_model = ChainCRF(config, params)

local x = torch.randn(config.batch_size,config.length,config.feature_size)
local log_edge_potentials_net = crf_model.log_edge_potentials_network
local log_edge_potentials_value = log_edge_potentials_net:forward(x)


local sum_product = SumProductInference(y_shape)
local predicted_edge_marginals, logZ = sum_product:infer_values(log_edge_potentials_value)
Inference1DUtil:assert_calibrated_edge_marginals(predicted_edge_marginals, y_shape) 
local predicted_node_marginals = Inference1DUtil:edge_marginals_to_node_marginals(predicted_edge_marginals, y_shape)
local function error_func(margs) return (margs - predicted_node_marginals):clone():abs():max() end


local unconstrained_optimization_config = 	{
	num_iterations =  30,
	learning_rate_scale = 1.0,
	momentum_gamma = 0.75,
	line_search = true,
	init_line_search_step = 1,
	rtol = 0.00000001,
	learning_rate_decay = 0.50,
	return_optimization_diagnostics = true
}

local constrained_optimization_config = Util:copyTable(unconstrained_optimization_config)
constrained_optimization_config.momentum_gamma = 0.66
constrained_optimization_config.line_search = false


local function select_prediction_tensor(out)
	if(shared_optimization_config.return_optimization_diagnostics) then
		return out[1]
	else
		return out
	end
end

-- Now do gradient-based mean-field
print('Gradient Descent')
local gd_config = {
	return_objective_value = true,
	entropy_weight = 1.0,
	logit_iterates = true,
	return_optimization_diagnostics = true,
	optimization_config = unconstrained_optimization_config
}

local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(crf_model)
local pred = gd_prediction_net:forward(x)
local gd_margs = pred[1]
local obj = pred[2]
print(pred[3][2])
print(predicted_node_marginals[1])
print(gd_margs[1])
local err = error_func(gd_margs)
assert(err < 0.05)
print(err)


if(config.domain_size == 2) then
	print('Projected Gradient Descent')
	gd_config = {
		return_objective_value = true,
		entropy_weight = 1.0,
		logit_iterates = false,
		mirror_descent = false,
		return_optimization_diagnostics = true,
		optimization_config = constrained_optimization_config
	}
	local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(crf_model)
	local pred = gd_prediction_net:forward(x)
	local gd_margs = pred[1]
	local obj = pred[2]
	print(gd_margs[5])
	print(predicted_node_marginals[5])
	local err = error_func(gd_margs)
	print(err)
	assert(err < 0.05)
end

print('KL Projected Gradient Descent')
gd_config = {
	return_objective_value = true,
	entropy_weight = 1.0,
	logit_iterates = false,
	mirror_descent = true,
	return_optimization_diagnostics = true,
	optimization_config = constrained_optimization_config
}
local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(crf_model)
local pred = gd_prediction_net:forward(x)
local gd_margs2 = pred[1]
local obj2 = pred[2]

local err = error_func(gd_margs2)
assert(err < 0.05)
print(err)

-- BCD mean-field
print('Mean Field')
local mf_config = {
	num_iters = 15
}
local mf_inference_net = MeanFieldInference(y_shape,mf_config):inference_net()
mf_node_marginals = mf_inference_net:forward(log_edge_potentials_value)
local err = error_func(mf_node_marginals)
assert(err < 0.05)
print(err)
