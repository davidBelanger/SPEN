local GradientDirection, parent = torch.class('nn.GradientDirection', 'nn.Container')

--This module wraps an a network (called energy_network below). In the forward pass, it returns the gradient of the energy_network.
--The backwards pass, which requires computing a Hessian-vector product, uses finite differences. See, for example, 
--Justin Domke "Generic Methods for Optimization-Based Modeling." AISTATS 2012 for  a derivation. 

--The energy_network expects a table of inputs, and the output of GradientDirection is the partial derivative 
-- of energy_network with respect to input[tableIndex]. 

--Since the Hessian-vector product is computed using finite differences, the approximation will 
-- will only be well-behaved if energy_network is smooth in both its inputs and parameters. Also, epsilon (the finite difference step size)
-- needs to be small. 

--NOTE: This does not expect data stored in energy_network will persist. All computations here copy things out of the return values from energy_network. 
--This means that if you have a bunch of GradientDirection modules, you can pass all of them the same energy_network, without cloning it

--NOTE: turning the inplace option on (for the sake of memory savings) is probably fine, but we haven't tested it thoroughly

--NOTE: this slightly breaks the API of nn.Module, since every call to updateGradInput also does accGradParameters (for efficiency reasons).
--This prevents any use case where you seek to update gradInput without changing gradparameters, though, so be careful.	

function GradientDirection:__init(energy_network,tableIndex,epsilon,inplace)
	parent.__init(self)
	self.index = index
	self.tableIndex = tableIndex
	self.singlePass = true --this is for efficiency. 
	assert(tableIndex)
	self.energy_network = energy_network
	self.epsilon = self.epsilon or 0.00001
	self.inplace = inplace or false

	table.insert(self.modules,energy_network) --this is so things like zeroGradParameters() will get called on the energy_network

	self.outputInitialized = false
	self.gradInitialized = false
end

function GradientDirection:updateOutput(input)
	local input_to_use = self.preprocess and self.preprocess:forward(input) or input

	local energy = self.energy_network:forward(input_to_use)
	self.fixedUnitGradient = self.fixedUnitGradient or energy:clone()
	self.fixedUnitGradient:resizeAs(energy):fill(1.0)
	self.energy_network:updateGradInput(input_to_use,self.fixedUnitGradient) --note: it's important not to call backward, since you don't want to acc grad parameters
	local output_ref =  self.preprocess and self.preprocess:updateGradInput(input_to_use,self.energy_network.gradInput)[self.tableIndex] or self.energy_network.gradInput[self.tableIndex] 
	
	if(not self.outputInitialized) then
		self.output =  Util:deep_clone(output_ref)
		self.outputInitialized = true
	end
	Util:deep_copy(self.output,output_ref)

	return self.output
end

function GradientDirection:genericBackward(backwardOperator,input,gradOutput,ugi)
	local raw_input = input
	local input_to_use = self.preprocess and self.preprocess:forward(input) or input
	local input_to_shift = input_to_use[self.tableIndex] 

	local gradOutput_to_use = gradOutput

	if(not self.gradInitialized) then 
		self.fdGradInput =   Util:deep_clone(input_to_use)
		self.shiftedInputs = Util:deep_clone(input_to_use)
		self.gradInitialized = true
	end

 	if(ugi) then
 		Util:deep_apply_inplace_two_arg(self.fdGradInput,input_to_use,function(s,t) s:resizeAs(t):zero() end)
	end

	Util:deep_copy(self.shiftedInputs,input_to_use)

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
	self.gradNormsForDenom:resizeAs(self.gradNorms):copy(self.gradNorms):add(1e-8) --to prevent 0/0

	
	self.gradOutputNormalized = self.gradOutputNorm or  Util:deep_clone(gradOutput_to_use)
	Util:deep_copy(self.gradOutputNormalized,gradOutput_to_use)
	
	local function expand_and_divide(t)
		local gSize = torch.LongStorage(t:dim()):fill(1)
		gSize[1] = self.gradNormsForDenom:size(1)
		assert(self.gradNormsForDenom:gt(0):all(),'nan denominiator: '..self.gradNormsForDenom:max()..' '..self.gradNormsForDenom:min()) --there's no way it would be less than 0. this checks for nans though
		local denom = self.gradNormsForDenom:view(gSize):expandAs(t)
		t:cdiv(denom)
	end
	Util:deep_apply_inplace(self.gradOutputNormalized,expand_and_divide)
	
	self.gradNorms:mul(1/(2*self.epsilon)) --this is the per-example scale for the finite difference approximation

	Util:deep_apply_inplace_two_arg(input_to_shift,self.gradOutputNormalized,function(t,s) t:add(self.epsilon,s) end)
	self.energy_network:forward(input_to_forward)
	backwardOperator(self.energy_network,input_to_forward,self.gradNorms) 
 
 	if(ugi) then
 		Util:deep_apply_inplace_two_arg(self.fdGradInput,self.energy_network.gradInput,function(t,s) t:add(s) end)
	end

	Util:deep_apply_inplace_two_arg(input_to_shift,self.gradOutputNormalized,function(t,s) t:add(-2*self.epsilon,s) end)
	self.gradNorms:mul(-1) --this is so the the following terms will get added to the finite difference approximation with a negative sign
	self.energy_network:forward(input_to_forward)
	backwardOperator(self.energy_network,input_to_forward,self.gradNorms) 

	if(ugi) then
 		Util:deep_apply_inplace_two_arg(self.fdGradInput,self.energy_network.gradInput,function(t,s) t:add(s) end)
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
	if(self.preprocess) then 
		self.gradInput = self.preprocess.gradInput 
	else 
		self.gradInput = self.fdGradInput 
	end

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



