local GradientBasedInference, parent = torch.class('GradientBasedInference')
require 'UnrolledGradientOptimizer'

--here, the energy net takes in a block of normalized iterate values and returns a number
--todo: eventually it would be good to support the use case that the energy_net takes in logits directly
function GradientBasedInference:__init(y_shape,config)
	assert(y_shape)
	assert(config)
	self.y_shape = y_shape
	self.config = config
	self.domain_size = y_shape[#y_shape]
	self.use_single_probs = (self.domain_size == 2) and (not config.logit_iterates) and not config.continuous_outputs
	assert(not (config.continuous_outputs and config.logit_iterates) )
	self.optimization_config = config.optimization_config
	self.continuous_outputs = config.continuous_outputs
	assert(self.optimization_config)
end

--TODO: have this network take y-x
function GradientBasedInference:loss_augmented_spen_inference(spen, cost_net)
	assert(not self.continuous_outputs,'not implemented for continuous outputs')
	--now, the features return {ygt,F(x)}
	local loss_augmented_features_network = nn.ParallelTable():add(nn.Identity()):add(spen.features_network)

	local y = nn.Identity()()
	local inputs_to_loss_augmented_energy = nn.Identity()()
	local y_ground_truth = nn.SelectTable(1)(inputs_to_loss_augmented_energy)

	local features = nn.SelectTable(2)(inputs_to_loss_augmented_energy)
	local model_energy = spen.energy_network({y,features})
	local loss_term = cost_net({y,y_ground_truth})         
	local loss_augmented_energy = nn.CSubTable()({model_energy,loss_term})

	local loss_augmented_energy_network = nn.gModule({y,inputs_to_loss_augmented_energy},{loss_augmented_energy})

	local initialization_network = nn.Sequential():add(nn.SelectTable(2)):add(spen.initialization_network)
	return self:full_inference_net(initialization_network,loss_augmented_features_network, loss_augmented_energy_network)
end

function GradientBasedInference:spen_inference(spen)
	return self:full_inference_net(spen.initialization_network,spen.features_network, spen.energy_network)
end

function GradientBasedInference:full_inference_net(initialization_network, features_network, energy_network)
	assert(initialization_network)
	assert(features_network)
	assert(energy_network)
	local input = nn.Identity()()
	local conditioning_values = features_network(input)
	local expanded_initialization_net = self:expand_initialization_net(initialization_network)
	local init_y = expanded_initialization_net(conditioning_values)
	local expanded_energy_net = self:expand_energy_net(energy_network)

	local return_modules = self:inference_net_helper(expanded_energy_net,init_y,conditioning_values)
	return nn.gModule({input}, return_modules)
end

function GradientBasedInference:expand_initialization_net(init_net)
	if(self.continuous_outputs) then return init_net end
	local expanded = nn.Sequential():add(init_net)
	if(self.use_single_probs) then 
		--todo: this won't work if we have a single continuous value
		expanded:add(nn.Narrow(#self.y_shape,self.domain_size,1))
	end
	if(self.config.logit_iterates) then 
		expanded:add(nn.Log())
	end

	return expanded
end

function GradientBasedInference:expand_energy_net(energy_net)
	if(self.continuous_outputs) then return energy_net end
	local config = self.config
	local use_single_probs = (self.domain_size == 2) and (not config.logit_iterates)
	if(config.optimization_config.line_search) then assert(config.logit_iterates) end

	local prepend_softmax_to_energy = config.logit_iterates

	local expand_iterates = use_single_probs --TODO: the energy could take in single probs as well.

	if(prepend_softmax_to_energy  or  expand_iterates) then
		local y = nn.Identity()()
		local conditioning_values = nn.Identity()()

		local y_to_energy
		if(prepend_softmax_to_energy) then
			y_to_energy = self:softmax_nd(self.y_shape)(y)
		else --todo: we don't support using single logits for binary problems
			y_to_energy = self:expand_to_probs(y)
		end

		local energy = energy_net({y_to_energy,conditioning_values})
		return nn.gModule({y,conditioning_values},{energy})
	else
		return energy_net
	end

end

function GradientBasedInference:inference_net_helper(energy_net, init_y, conditioning_values)
	local include_entropy_term = not self.config.mirror_descent

	local objective_for_evaluation = energy_net
	local energy_net_with_entropy
	if(self.config.entropy_weight ~= 0.0) then
		energy_net_with_entropy = self:subtract_entropy_from_objective(energy_net)
		objective_for_evaluation = energy_net_with_entropy
	end

	local objective_for_optimization
	if(self.config.mirror_descent or self.config.entropy_weight == 0) then
		objective_for_optimization = energy_net
	else
		objective_for_optimization = energy_net_with_entropy
	end

	local optimization_config = self.optimization_config
	optimization_config.iterate_shape = self:iterate_shape()

	optimization_config.return_all_iterates = self.config.return_all_iterates
	if(self.config.return_optimization_diagnostics) then
		--todo: don't compute the objective value necessarily...
		self.config.return_objective_value = true
		optimization_config.return_all_iterates = true
		optimization_config.return_objective_values = true
		optimization_config.return_convergence_indicators = true
	end


	if((not self.config.logit_iterates) and (not self.config.continuous_outputs)) then
		if(not self.config.mirror_descent) then
			assert(self.domain_size == 2)
			eps = 1e-3 --we clamp to the interior of [0,1]
			--TODO: we shouldn't have to be clamping so aggressively. If we don't, sometimes things diverge if we have an entropy term in the objective, though.
			optimization_config.projection_function = function(t) return nn.Clamp(eps,1-eps)(t) end
		else
			if(self.domain_size == 2) then
				optimization_config.custom_gradient_step = function(point, direction, learning_rate) return self:binary_entropic_mirror_descent_update(point,direction,learning_rate,self.config.entropy_weight) end
			else
				optimization_config.custom_gradient_step = function(point, direction, learning_rate) return self:entropic_mirror_descent_update(point,direction,learning_rate,self.config.entropy_weight) end
			end

			optimization_config.line_search = false

		end
	end

	local optimization_output = unrolled_gradient_descent_optimizer(objective_for_optimization, optimization_config)({init_y,conditioning_values})
	local raw_prediction
	if(optimization_config.return_all_iterates) then
		raw_prediction = nn.SelectTable(1)(optimization_output)
	else
		raw_prediction = optimization_output
	end
	local prediction = raw_prediction
	if(self.config.logit_iterates) then
		prediction = self:softmax_nd(self.y_shape)(raw_prediction) 
	elseif(self.use_single_probs) then
		prediction = self:expand_to_probs(raw_prediction)
	end
	local to_return = {prediction}

	if(self.config.return_all_iterates) then
		local all_iterates = nn.SelectTable(2)(optimization_output)
		if(self.config.logit_iterates) then
			local all_iterate_shape = {optimization_config.num_iterations}
			for i = 1,#self.y_shape do
				table.insert(all_iterate_shape,self.y_shape[i])
			end
			all_iterates = self:softmax_nd(all_iterate_shape)(all_iterates) 
		elseif(self.use_single_probs) then
			assert(false,'not implemented')
		end
		table.insert(to_return,all_iterates)
	end
	if(self.config.return_objective_value) then
		--todo: we could actually be grabbing the value from the optimization_output if it's computing the objectives. 
		assert(false,'need to be sharing parameters when cloning')
	 	local objective = objective_for_evaluation:clone()({raw_prediction, conditioning_values}) 
	 	table.insert(to_return,objective)
	end

	if(self.config.return_optimization_diagnostics) then
		for i = 2,4 do
			table.insert(to_return,nn.SelectTable(i)(optimization_output))
		end
	end

	return to_return
end

function GradientBasedInference:iterate_shape()
	if(self.use_single_probs) then
		local it_shape = Util:copyTable(self.y_shape)
		it_shape[#it_shape] = 1
		return it_shape
	else
		return self.y_shape
	end
end

function GradientBasedInference:softmax_nd(shape)
	assert(not self.continuous_outputs,'should not be here')
	local s = nn.Sequential()
	local big_batch_shape = 1
	for i = 1,(#shape-1) do
		big_batch_shape = big_batch_shape*shape[i]
	end
	s:add(nn.View(big_batch_shape,self.domain_size))
	s:add(nn.SoftMax())
	s:add(nn.View(unpack(shape)))

	return s
end

--typically, you add entropy to the objective being maximized, so here, it gets subtracted because we minimize the negative objective
function GradientBasedInference:subtract_entropy_from_objective(objective_net)
	assert(self.config.entropy_weight > 0)
	assert(not self.continuous_outputs,'should not be using entropy smoothing for continuous output problems')
	local y = nn.Identity()()
	local conditioning_values = nn.Identity()()
	local y_node_margs
	if(self.config.logit_iterates and not self.use_single_probs) then
		y_node_margs = self:softmax_nd(self.y_shape)(y)
	elseif(self.use_single_probs) then
		y_node_margs = self:expand_to_probs(y)
	else
		y_node_margs = y
	end

	local entropy_term = nn.MulConstant(-self.config.entropy_weight)(nn.Entropy()(y_node_margs)) 

	local objective = objective_net({y,conditioning_values})
	objective = nn.CAddTable()({objective,entropy_term})

	return nn.gModule({y,conditioning_values},{objective})
end


function GradientBasedInference:expand_to_probs(y)
	local one_minus_y = nn.AddConstant(1,true)(nn.MulConstant(-1,false)(y)) --it's very important to not do the first MulConstant in place. Otherwise, y will get effected by this math.
	local ndims = #self.y_shape
	return nn.JoinTable(ndims)({one_minus_y,y})
end


function GradientBasedInference:entropic_mirror_descent_update(y,direction,learning_rate,entropy_weight)
	local epsilon = 0.0000001

	assert(learning_rate > 0)
	local scaled_grad = nn.MulConstant(-learning_rate)(direction)


	local logits = nn.Log()(nn.AddConstant(epsilon)(y))
	logits = nn.CAddTable()({logits,scaled_grad})

	if(entropy_weight > 0.0) then
		--todo: should there be a d here or somethng? We needed 2 to get it to work for binary emd...
         invTemperature = 1/(1 + learning_rate * entropy_weight)
         logits = nn.MulConstant(invTemperature)(logits)
	end

	local logZ = nn.LogSumExp()(logits)

	local shape = Util:copyTable(self.y_shape)
	shape[#shape]  = 1
	local logZ_expand = nn.Replicate(self.domain_size,#self.y_shape,#self.y_shape)(nn.View(unpack(shape))(logZ)) --todo: ideally we'd make this more generic
	local logProbs = nn.CSubTable()({logits,logZ_expand})
	return nn.Exp()(logProbs)
end


function GradientBasedInference:binary_entropic_mirror_descent_update(y,direction,learning_rate,entropy_weight)
	local epsilon = 0.0000001
	assert(learning_rate > 0)
	local one_minus_y = nn.AddConstant(1+epsilon,true)(nn.MulConstant(-1,false)(y)) --it's very important to not do the first MulConstant in place. Otherwise, y will get effected by this math.
	--todo: is it better to take the log of the ratio, or the difference of logs?
	local ratio = nn.CDivTable()({y,one_minus_y})
	local log_ratio = nn.Log()(nn.AddConstant(epsilon,true)(ratio))

	local scaled_grad = nn.MulConstant(-2*learning_rate)(direction)
	local logit_step = nn.CAddTable()({log_ratio,scaled_grad})

	if(entropy_weight ~= 0.0) then
		invTemperature = 1/(1 + 2 * learning_rate * entropy_weight)
		logit_step_with_temp = nn.MulConstant(invTemperature,true)(logit_step)
	else
		logit_step_with_temp = logit_step
	end

	return nn.Sigmoid()(logit_step_with_temp)
end