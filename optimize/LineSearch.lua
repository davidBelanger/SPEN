local LineSearch, parent = torch.class('nn.LineSearch', 'nn.Container')

-- Basic back-tracking line search.
-- The input objective_function expects {objective_function_inputs,direction} as input and outputs an alpha, 
-- which will be negative, since it steps in the negative gradient direction. 
-- It returns zero for the gradient of alpha with respect to everything. Obviously this isn't correct at all points.
-- We check that 
-- f(x + alpha * p) < f(x)
-- where alpha is the step size and p is the step direction. We evaluate at a small 
-- set of candidate alphas, defined by backtracking with a power of two. The only time the gradient of alpha with respect to the 
-- f or p would not be true is if 
-- The only time it would not be true is if the above condition holds with equality, which won't happen for most of the small set of alphas we try. 

-- The input init_step is the first step length to try (before backtracking). This quantity should be
-- *positive*. 

--  Line search only shifts the iterate, while the objective_function may take more inputs. 
-- shift_inputs is a function that it knows how to do the shifting. 

--TODO: this could take the value evaluated at iterate as an input, as this might have been precomputed elsewhere
--TODO: this needs to be minibatched. needs to return a different step per batch element!
function LineSearch:__init(objective_function, init_step, shift_inputs)
	self.objective_function = objective_function
	self.init_step = init_step
	self.shift_inputs = shift_inputs
	self.modules = {}
end


--Inputs: {objective_inputs, direction}
function LineSearch:updateOutput(input)
	if(not self.objective_inputs) then
		self.objective_inputs = Util:deep_clone(input[1])
	end

	Util:deep_copy(self.objective_inputs,input[1])

	local direction = input[2]
	
	local objective_function
	batch_size = direction:size(1)
	--TODO: don't re-allocate alpha
	self.alpha = self.alpha or direction.new():resize(batch_size)
	self.alpha:fill(-self.init_step) --negate because you step in the negative gradient direction

	local value0_output = self.objective_function:forward(self.objective_inputs)
	
	if(not self.value0) then
		self.value0 = value0_output:clone()
	end
	local iterate_shape = direction:size():totable()
	self.value0:copy(value0_output)
	self.prev_alpha = self.prev_alpha or self.alpha:clone()
	local num_backtracks = 12
	i = 1

	self.offsets = self.offsets or direction:clone()
	--This maintains a different step size for every element of the minibatch
	while(i <= num_backtracks) do
		if(i == 1) then
			self.shift_amount = self.shift_amount or self.alpha:clone()
			self.shift_amount:copy(self.alpha)
		else
			self.shift_amount:copy(self.alpha):add(-1,self.prev_alpha)
		end

	   	local amount = Util:expand_to_shape(self.shift_amount,iterate_shape)
	   	self.offsets:copy(direction):cmul(amount)
		self.shift_inputs(self.objective_inputs,self.offsets) 

		local value = self.objective_function:forward(self.objective_inputs)

		local convergence_pattern = self:satisfactory(self.value0,value)
		--TODO: we could break if most, but not all, of them have converged
		if(convergence_pattern:all()) then
			break
		else
			--TODO: we could remvoe the memory allocations here
			self.prev_alpha:copy(self.alpha)
			local elements_to_change = convergence_pattern:lt(1)
			local denominator = elements_to_change:typeAs(self.alpha):add(1)
			self.alpha:cdiv(denominator)
		end
		i = i + 1
	end

	--If it couldn't find a valid step,then don't step at all. 
	if(i > num_backtracks) then
		self.alpha:zero()
	end
	--assert(i <= num_backtracks,"backtracking line search did not find a sufficient step. Are you sure the gradient was correct?")
	self.output = self.alpha
	return self.output
end

function LineSearch:satisfactory(value0,value)
	--TODO: we could check for more sophisticated conditions. These will likely depend on gradients, though...
	--TODO: remove the memory allocation
	return value:lt(value0)
end

function LineSearch:updateGradInput(input,gradOutput)
	if(not self.gradInput) then
		self.gradInput = Util:deep_clone(input)
	end
	Util:deep_apply(self.gradInput,function(v) return v:zero() end) --TODO: really, this only needs to be done once

	return self.gradInput

end