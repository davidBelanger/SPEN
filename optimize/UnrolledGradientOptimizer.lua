local UnrolledGradientOptimizer = torch.class('UnrolledGradientOptimizer')

require 'LineSearch'
require 'GradientDirection'
require 'Cond'
require 'Predicate'
-- This takes {conditioning_values, iterate, hid_state} and returns these at the next timestep
-- We assume that the inputs to the objective are {iterate, conditioning_values} (in that order)
-- Here, the hid_state depends on the algorithm, and may be null. For gradient descent with momentum,
-- It is the momentum. 

--this takes nngraph Nodes and returns nngraph Nodes
function get_optimization_step(inputs,objective,t,config)
	local conditioning_values = inputs.conditioning_values
	local yt = inputs.iterate
	local ht = inputs.hid_state
	local is_converged = inputs.is_converged
  	local current_objective = inputs.objective

  	if(config.line_search) then config.momentum_gamma = 0 end

  	local cell = get_recurrent_cell(objective, t, config)

  	--cell and no_op. 
  	--Input:
	--{yt,ht,conditioning_values, objective}

	--Return:
  	--{yt1,ht1,converged, objective} or {yt1,ht1,converged}


  	cell = nn.Sequential():add(nn.NarrowTable(1,3)):add(cell)

  	local noop_net = nn.ParallelTable()
  	noop_net:add(nn.Identity())
  	noop_net:add(nn.Identity())
  	noop_net:add(nn.Constant(torch.ByteTensor(1):fill(1)))
  	noop_net:add(nn.Identity())
	local inputs_to_cell = nn.Identity()({yt,ht,conditioning_values, current_objective})
	local cond_step = nn.Cond(noop_net,cell)({is_converged,inputs_to_cell})
	local yt1 = nn.SelectTable(1)(cond_step)

	local outputs = {
		conditioning_values = inputs.conditioning_values,
		iterate = yt1,
	}

	if(t < config.num_iterations) then
		outputs.hid_state = nn.SelectTable(2)(cond_step)
		outputs.is_converged = nn.SelectTable(3)(cond_step)
	end
	if(config.return_objective_values) then 
		outputs.objective = nn.SelectTable(4)(cond_step)
	else
		outputs.objective = current_objective
	end

	return outputs
end

--this instantiates a network (not an nngraph node) for taking a single step. 
--its inputs are {yt,ht,conditioning_values}
--its outputs are {yt1,ht1,converged,objective (optional)}. The outputs may also contain a fourth entry for the objective value.

function get_recurrent_cell(objective,t,config)
	local yt = nn.Identity()()
	local ht = nn.Identity()()
	local conditioning_values = nn.Identity()()

	local learning_rate = config.learning_rate_scale/math.pow(1 + config.learning_rate_decay*t, config.learning_rate_power) 

	local grad_data = nn.GradientDirection(objective,1,config.return_objective_values,config.finite_difference_step,false)({yt,conditioning_values})

	--grad_data = nn.PrintNoNewline(false,true,'gd',function(t) io.write(t:norm().." ") end)(grad_data)
	local grad
	local objective_value
	if(config.return_objective_values) then
		grad = nn.SelectTable(1)(grad_data)
		objective_value = nn.SelectTable(2)(grad_data)
	else
		grad = grad_data
	end

	local ht1, direction
	if(config.momentum_gamma > 0) then
		--assert(not config.line_search,"shouldn't use momentum with line search")
		--ht1 = gamma*ht + (1- gamma)*grad_t
		local weight
		if(t > 1) then
			weight = config.momentum_gamma
		else
			weight = 0
		end
		local function mul_layer(weight,in_place)
			if(config.learn_hyperparams) then
				local mul = nn.Mul()
				mul.weight:fill(weight)
				return mul
			else
				return nn.MulConstant(weight,in_place)
			end
		end
		--NOTE: this does a little bit of wasted computation at t = 1, but it's best to multiply ht by 0, since otherwise the input ht doesn't appear in the graph.
		local term1 = mul_layer(weight,t > 1)(ht) --if t ==1, then can't do in place, because the backwards pass in MulConstant would divide by 0
		local term2 = mul_layer(1 - weight,true)(grad)
		ht1 = nn.CAddTable()({term1,term2})
		direction = ht1
	else
		ht1 = ht
		direction = grad
	end

	local function gradient_update(point,direction,lr) 
		if(config.custom_gradient_step) then
			assert(not config.line_search,"Custom gradient steps like mirror descent are probably incompatible with line search.")
			return config.custom_gradient_step(point,direction,lr)
		end

       local scaled_direction 
       assert(lr > 0)
       if(config.learn_hyperparams) then
       		error('here')
       		assert(not config.line_search,"shouldn't be learning learning rates if doing line search")
       	   --TODO: this doesn't ensure that the LR is positive. need to use a exp or something applied to a nn.Constant
           scaled_direction = nn.Mul()(direction)
           local weight = scaled_direction.data.module.weight
           weight:fill(-lr)
       elseif(config.line_search) then
       	  assert(config.iterate_shape)
       		-- The inputs to the objective are {iterate, conditioning_values}. Line search only shifts the iterate.
       		-- We pass this local function to LineSearch so that it knows how to do the shifting. 
	   		local function shift_inputs(point,shift)
	   			point[1]:add(shift)
	   		end
	   	  inputs_to_objective_function = nn.ParallelTable():add(nn.Identity()):add(nn.Identity())({yt,conditioning_values}) --todo: there's probably a cleaner way to do this
       	  step_sizes = nn.LineSearch(objective, config.init_line_search_step, shift_inputs)({inputs_to_objective_function,direction})
       	  expanded_step_sizes = Util:expand_to_shape_nn(step_sizes,config.iterate_shape)
       	  scaled_direction = nn.CMulTable()({expanded_step_sizes, direction})
       else
           scaled_direction = nn.MulConstant(-lr,true)(direction)
       end
	    return nn.CAddTable()({point,scaled_direction})
	end


    local yt1 = gradient_update(yt, direction, learning_rate)
   	local converged = converged_iterates(yt,yt1,config.rtol,config.iterate_shape)


	if(config.projection_function) then
		yt1 = config.projection_function(yt1)
	end

	local values_to_return = {yt1,ht1,converged}

	if(config.return_objective_values) then
		table.insert(values_to_return,objective_value)
	end

	return nn.gModule({yt,ht,conditioning_values},values_to_return)

end

function converged_iterates(yt,yt1,rtol,shape)
	--todo: don't need to reallocate memory here
	local function predicate(a,b)
		return ((a - b):norm() / a:norm()) < rtol
	end
	return nn.Predicate(predicate)({yt,yt1})
end

function initialize_hidden_state(init_iterate, config)
	if(config.momentum_gamma > 0) then
		return nn.MulConstant(0.0)(init_iterate)
	else
		return nn.Constant(torch.zeros(1))(init_iterate)
	end
end

--This returns a module that takes {conditioning_values, init_iterate} and returns optimized_iterate
-- If config.return_objective_values, it also returns a packed tensor of the history of objective values over time. 
-- If config.return_all_iterates, it also returns a packed tensor of all of the iterates over time.
-- If config.return_convergence_indicators, it also returns a packed tensor of the indicators for whether optimization at converged yet at timestep t. This is *not* per-minibatch-element convergence. 

function unrolled_gradient_descent_optimizer(objective, input_config)

	local input_config = input_config or {}

	local config = {
		iterate_shape = input_config.iterate_shape or nil, --this needs to be non-nil if return_all_iterates is true
		return_all_iterates = input_config.return_all_iterates or false,
		return_objective_values = input_config.return_objective_values or false,
		return_convergence_indicators = input_config.return_convergence_indicators or false,
		finite_difference_step = input_config.finited_difference_step or 0.000001,
		num_iterations      = input_config.num_iterations or 10,
		learning_rate_scale = input_config.learning_rate_scale or 0.01,
		learning_rate_decay = input_config.learning_rate_decay or 0, -- set to 0.0 to not decay learning rate
		learning_rate_power = input_config.learning_rate_power or 0.5,
		learn_hyperparams   = input_config.learn_hyperparams or false,
		momentum_gamma      = input_config.momentum_gamma or 0.0, -- set to 1.0 to not use momentum
		projection_function = input_config.projection_function or nil, -- set to function(x) nn.Clamp(0,1)(x) end
		line_search = input_config.line_search or false,
		init_line_search_step = input_config.init_line_search_step or 1.0,
		rtol = input_config.rtol or 0.0001,
		custom_gradient_step = input_config.custom_gradient_step or nil
	}

	if(config.return_all_iterates) then assert(config.iterate_shape) end

	local conditioning_values = nn.Identity()()
	local init_iterate = nn.Identity()()

	local initial_data = {
		conditioning_values=conditioning_values,
		iterate = init_iterate,
		hid_state = initialize_hidden_state(init_iterate, config),
		is_converged = nn.Constant(torch.ByteTensor(1):fill(0))(init_iterate),
		objective = nn.Constant(torch.Tensor(config.iterate_shape[1]):fill(100000000))(init_iterate)

	}
	local curr_step = initial_data
	local objective_values = {}
	local iterates = {}
	local convergence_indicators = {}
	for i = 1,config.num_iterations
 	do		
 		curr_step = get_optimization_step(curr_step, objective, i, config)

		if(config.return_objective_values) then
			table.insert(objective_values, nn.View(config.iterate_shape[1],1)(curr_step.objective))
		end
		if(config.return_all_iterates) then
			iterate_rank = table.getn(config.iterate_shape)
			size = torch.LongStorage(iterate_rank + 1)
			for i = 1,iterate_rank do
				size[i] = config.iterate_shape[i]
			end
			size[iterate_rank + 1] = 1
			table.insert(iterates,nn.View(size)(curr_step.iterate))
		end
		if(config.return_convergence_indicators) then
			table.insert(convergence_indicators,curr_step.is_converged)
		end
	end

	--NOTE: this doesn't compute the objective at the final iterate. not sure how best to do this

	local final_iterate = curr_step.iterate
	local tensors_to_return = {final_iterate}
	if(config.return_objective_values) then
		local objective_values_tensor = nn.JoinTable(2,2)(objective_values)
		table.insert(tensors_to_return,objective_values_tensor)
	end

	if(config.return_all_iterates) then
		local size = table.getn(config.iterate_shape) + 1
		local all_iterates = nn.JoinTable(size, size)(iterates)
		table.insert(tensors_to_return,all_iterates)
	end

	if(config.return_convergence_indicators) then
		local all_indicators = nn.JoinTable(1)(convergence_indicators)
		table.insert(tensors_to_return,all_indicators)
	end

	return nn.gModule({init_iterate,conditioning_values},tensors_to_return)

end








