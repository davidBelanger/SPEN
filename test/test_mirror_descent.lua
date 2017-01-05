require 'Imports'



--This tests GradientBasedInference. It constructs a simple objective function of linear + entropy, which can be solved in closed form with a softmax. 
--It makes sure that GradientBasedInference works for mirror descent, unconstrained GD on logits with line search, and unconstrained GD on logits w/o line search.

local batch_size = 40
local in_size = 5
local domain_size = 10
local y_shape = {batch_size,domain_size}

local linear_model = nn.Linear(in_size,domain_size)
linear_model.weight:mul(1.5)

local feed_forward_net = nn.Sequential():add(linear_model):add(nn.SoftMax())

local scores = nn.Identity()()
local y = nn.Identity()()
local neg_scores = nn.MulConstant(-1)(scores)
local energy = nn.Sum(2)(nn.CMulTable()({neg_scores,y}))
local energy_net = nn.gModule({y,scores},{energy})

local spen = {
	initialization_network = nn.Constant(torch.ones(batch_size,domain_size):fill(1.0/domain_size)),
	features_network = linear_model,
	energy_network = energy_net
}

local function assert_correct(prediction_net)
	for i = 1,10 do
		local x = torch.randn(batch_size,in_size)
		local f1 = feed_forward_net:forward(x)
		local f2 = prediction_net:forward(x)
		print(f1)
		print(f2)
		assert((f1 - f2):norm()/f1:norm() < 0.05)
	end
end

local unconstrained_optimization_config = 	{
	num_iterations =  30,
	learning_rate_scale = 1.0,
	momentum_gamma = 0.9,
	line_search = false,
	init_line_search_step = 1,
	rtol = 0.0001,
	learning_rate_decay = 0,
	return_optimization_diagnostics = false
}
local gd_config = {
	return_objective_value = false,
	entropy_weight = 1.0,
	logit_iterates = false,
	mirror_descent = true,
	return_optimization_diagnostics = false,
	optimization_config = unconstrained_optimization_config
}
local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(spen)

assert_correct(gd_prediction_net)

print('----------------------------')

local unconstrained_optimization_config = 	{
	num_iterations =  30,
	learning_rate_scale = 1.0,
	momentum_gamma = 0.5,
	line_search = true,
	init_line_search_step = 1,
	rtol = 0.0001,
	learning_rate_decay = 0,
	return_optimization_diagnostics = false
}

local gd_config = {
	return_objective_value = false,
	entropy_weight = 1.0,
	logit_iterates = true,
	mirror_descent = false,
	return_optimization_diagnostics = false,
	optimization_config = unconstrained_optimization_config
}
local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(spen)

assert_correct(gd_prediction_net)

print('----------------------------')

local unconstrained_optimization_config = 	{
	num_iterations =  30,
	learning_rate_scale = 1.0,
	momentum_gamma = 0.5,
	line_search = false,
	init_line_search_step = 1,
	rtol = 0.0001,
	learning_rate_decay = 0.01,
	return_optimization_diagnostics = false
}

local gd_config = {
	return_objective_value = false,
	entropy_weight = 1.0,
	logit_iterates = true,
	mirror_descent = false,
	return_optimization_diagnostics = false,
	optimization_config = unconstrained_optimization_config
}
local gd_prediction_net = GradientBasedInference(y_shape, gd_config):spen_inference(spen)

assert_correct(gd_prediction_net)

