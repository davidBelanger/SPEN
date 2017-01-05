require 'Imports'

package.path = package.path .. ';model/?.lua'
require 'ChainSPEN'

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
config.energy_hid_size = 5
config.local_potentials_scale = 1.0
config.pairwise_potentials_scale = 1.0
config.data_independent_joint_energy = false

local spen_model = ChainSPEN(config)

local x = torch.randn(config.batch_size,config.length,config.feature_size)

local unconstrained_optimization_config = 	{
	num_iterations =  30,
	learning_rate_scale = 0.5,
	momentum_gamma = 0,
	line_search = true,
	init_line_search_step = 1,
	rtol = 0.0001,
	learning_rate_decay = 0.50,
	return_optimization_diagnostics = true
}

local constrained_optimization_config = Util:copyTable(unconstrained_optimization_config)
constrained_optimization_config.momentum_gamma = 0.75
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

local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(spen_model)

local pred = gd_prediction_net:forward(x)
local gd_margs = pred[1]
local obj = pred[2]
local function relative_error(t) return ((t - gd_margs):norm()/gd_margs:norm()) end


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
	local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(spen_model)
	local pred = gd_prediction_net:forward(x)
	local gd_margs = pred[1]
	print(gd_margs[1])

	local obj = pred[2]
	local err = relative_error(gd_margs)
	assert(err < 0.05)
	print(err)
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
local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(spen_model)
local pred = gd_prediction_net:forward(x)
local gd_margs = pred[1]
print(gd_margs[1])

local obj2 = pred[2]

local err = relative_error(gd_margs)
assert(err < 0.05)
print(err)
