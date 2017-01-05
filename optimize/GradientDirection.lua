local GradientDirection, parent = torch.class('nn.GradientDirection', 'nn.Container')

--it expects a table of inputs, and its output is the partial derivative of objective_function with respect to input[tableIndex]. 

--The gradient of this module (which corresponds to a Hessian-vector product) is computed using finite differences. 
--This will only be well-behaved if objective_function is smooth in both its inputs and parameters (and epsilon is small).

--NOTE: This does not expect data stored in objective_function will persist. All computations here copy things out of the return values from objective_function. 
--This means that if you have a bunch of GradientDirection modules, you can pass all of them the same objective_function, without cloning it

--NOTE: turning the inplace option on (for the sake of memory savings) is probably fine, but we haven't tested it thoroughly

--NOTE: this slightly breaks the API of nn.Module, since every call to updateGradInput also does accGradParameters (for efficiency reasons).
--This prevents any use case where you seek to update gradInput without changing gradparameters, though, so be careful.	

function GradientDirection:__init(objective_function,tableIndex,return_function_value,epsilon,inplace)
	parent.__init(self)
	self.index = index
	self.tableIndex = tableIndex
	self.singlePass = true --this is for efficiency. 
	assert(tableIndex)
	self.objective_function = objective_function
	self.return_function_value = return_function_value
	self.epsilon = self.epsilon or 0.00001
	self.inplace = inplace or false

	table.insert(self.modules,objective_function) --this is so things like zeroGradParameters() will get called on the objective_function

	self.outputInitialized = false
	self.gradInitialized = false
end

function GradientDirection:updateOutput(input)
	--we assume all interaction with the objective_function is stateless, so we need to do a forward pass right before doing a backwards pass
	local energy = self.objective_function:forward(input)
	self.fixedUnitGradient = self.fixedUnitGradient or energy:clone()
	self.fixedUnitGradient:resizeAs(energy):fill(1.0)

	--note: it's important not to call backward here, since you don't want to acc grad parameters.
	self.objective_function:updateGradInput(input,self.fixedUnitGradient) 
	local output_ref =  self.objective_function.gradInput[self.tableIndex] 
	
	if(not self.outputInitialized) then
		self.gradient_output =  Util:deep_clone(output_ref)
		
		if(self.return_function_value) then 
			self.value_output = energy:clone()
			self.output = {self.gradient_output, self.value_output}
		else
			self.output = self.gradient_output
		end

		self.outputInitialized = true
	end

	Util:deep_copy(self.gradient_output,output_ref)

	if(self.return_function_value) then
		self.value_output:copy(energy)
	end
	return self.output
end

function GradientDirection:genericBackward(backwardOperator,input,gradOutput,update_grad_input)
	local raw_input = input
	local input_to_shift = input[self.tableIndex] 


	if(not self.gradInitialized) then 
		self.fdGradInput =   Util:deep_clone(input)
		self.shiftedInputs = Util:deep_clone(input)
		self.gradInitialized = true
	end

 	if(update_grad_input) then
 		Util:deep_apply_inplace_two_arg(self.fdGradInput,input,function(s,t) s:resizeAs(t):zero() end)
	end

	local gradOutput_to_use
	if(self.return_function_value) then
		gradOutput_for_function_value = gradOutput[2]

		self.objective_function:forward(input)
		backwardOperator(self.objective_function,input,gradOutput_for_function_value) 
		Util:deep_apply_inplace_two_arg(self.fdGradInput,self.objective_function.gradInput,function(t,s) t:add(s) end)

		gradOutput_to_use = gradOutput[1]
	else
		gradOutput_to_use = gradOutput
	end
	assert(gradOutput_to_use:eq(gradOutput_to_use):all(),'nan gradOutput: '..gradOutput_to_use:max()..' '..gradOutput_to_use:min()) 


	Util:deep_copy(self.shiftedInputs,input)

	local input_to_shift = self.shiftedInputs[self.tableIndex]
	local input_to_forward = self.shiftedInputs

	self.gradNormsTable = self.gradNormsTable or Util:deep_apply(gradOutput_to_use,function(v) return v.new():resize(v:size(1)) end)

	local gradReshape = Util:deep_apply(gradOutput_to_use,function(v) return v:view(v:size(1),v:nElement()/v:size(1)) end)   --this doesn't copy any data
	Util:deep_apply_inplace_two_arg(self.gradNormsTable,gradReshape,function(s,t) torch.norm(s,t,2,2) end)
	Util:deep_apply_inplace_two_arg(self.gradNormsTable,gradReshape,function(s,t) torch.norm(s,t,2,2) end)

	local exampleTensor = Util:find_first_tensor(gradOutput_to_use)
	local minibatchSize =  exampleTensor:size(1) 

	--todo: all of this could be made more simple, and perhaps more efficient, for the special case that we don't have table-structured data. 
	self.gradNorms = self.gradNorms or exampleTensor.new()
	self.gradNorms:resize(minibatchSize)
	self.reducer = self.reducer or nn.Sequential():add(nn.CAddTable()):add(nn.Sqrt())
	if((not self.reducerIsCuda) and exampleTensor:type() == "torch.CudaTensor") then 
		self.reducer:cuda() 
		self.reducerIsCuda = true
	end

	Util:deep_map_reduce(self.gradNorms,self.gradNormsTable,function (v) return v:pow(2) end, function(result,tab) return result:copy(self.reducer:forward(tab)) end)
	self.gradNormsForDenom = self.gradNormsForDenom or self.gradNorms:clone()
	self.gradNormsForDenom:resizeAs(self.gradNorms):copy(self.gradNorms):cmax(1e-8) --to prevent 0/0

	
	self.gradOutputNormalized = self.gradOutputNorm or  Util:deep_clone(gradOutput_to_use)
	Util:deep_copy(self.gradOutputNormalized,gradOutput_to_use)
	
	local function expand_and_divide(t)
		local gSize = torch.LongStorage(t:dim()):fill(1)
		gSize[1] = self.gradNormsForDenom:size(1)
		assert(self.gradNormsForDenom:eq(self.gradNormsForDenom):all(),'nan denominiator: '..gradOutput_to_use:max()..' '..gradOutput_to_use:min()) 
		local denom = self.gradNormsForDenom:view(gSize):expandAs(t)
		t:cdiv(denom)
	end
	Util:deep_apply_inplace(self.gradOutputNormalized,expand_and_divide)
	
	self.gradNorms:mul(1/(2*self.epsilon)) --this is the per-example scale for the finite difference approximation

	Util:deep_apply_inplace_two_arg(input_to_shift,self.gradOutputNormalized,function(t,s) t:add(self.epsilon,s) end)
	self.objective_function:forward(input_to_forward)
	backwardOperator(self.objective_function,input_to_forward,self.gradNorms) 
 
 	if(update_grad_input) then
 		Util:deep_apply_inplace_two_arg(self.fdGradInput,self.objective_function.gradInput,function(t,s) t:add(s) end)
	end

	Util:deep_apply_inplace_two_arg(input_to_shift,self.gradOutputNormalized,function(t,s) t:add(-2*self.epsilon,s) end)
	self.gradNorms:mul(-1) --this is so the the following terms will get added to the finite difference approximation with a negative sign
	self.objective_function:forward(input_to_forward)
	backwardOperator(self.objective_function,input_to_forward,self.gradNorms) 

	if(update_grad_input) then
 		Util:deep_apply_inplace_two_arg(self.fdGradInput,self.objective_function.gradInput,function(t,s) t:add(s) end)
	end

	return nil
end

function GradientDirection:updateGradInput(input,gradOutput)
	local function operator(module,input,gradOutput) 
		module:updateGradInput(input,gradOutput) 
		if(self.singlePass) then
			module:accGradParameters(input,gradOutput,1.0)
		end
		return nil
	end
	self:genericBackward(operator,input,gradOutput,true)

	self.gradInput = self.fdGradInput 

	return self.gradInput
end

function GradientDirection:accUpdateGradParameters(input,gradOutput,lr)
	error('do not use this')
end

function GradientDirection:accGradParameters(input,gradOutput,lr)
	if(self.singlePass) then
		assert(lr == 1.0,'single pass mode is only setup to handle lr = 1.0')
	else
		local function operator(module,input,gradOutput) 	
			 module:updateGradInput(input,gradOutput) 		
			 module:accGradParameters(input,gradOutput,lr)
			return nil
		end
		self:genericBackward(operator,input,gradOutput,false)
	end
	return nil
end

function GradientDirection:clearState()

end



