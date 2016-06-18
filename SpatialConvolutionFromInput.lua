local SpatialConvolutionFromInput, parent = torch.class('nn.SpatialConvolutionFromInput', 'nn.Container')


--NOTE: this assumes that the convolution is done with zero bias

function SpatialConvolutionFromInput:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
	parent.__init(self)
	dW = dW or 1
   	dH = dH or 1
   	padW = padW or 0
   	padH = padH or 0

	self.nInputPlane = nInputPlane
	self.nOutputPlane = nOutputPlane
	assert(self.nInputPlane == 1)
	assert(self.nOutputPlane == 1)
	self.kW = kW
	self.kH = kH
	self.conv_module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
	self.conv_module.bias:zero()
	table.insert(self.modules,self.conv_module)
end

function SpatialConvolutionFromInput:parameters()
	return nil
end

function SpatialConvolutionFromInput:updateOutput(input)
	assert(#input == 2)
	local data = input[1]
	local kernel = input[2]

	for i = 1,data:size(1) do
		self.conv_module.weight:copy(kernel:select(1,i):expandAs(self.conv_module.weight)) --todo: could probably just copy the reference
		self.conv_module:updateOutput(data:select(1,i))
		local tmp_output = self.conv_module.output
		if(i == 1) then
			self.output = self.output or input.new()
			self.output:resize(data:size(1),tmp_output:size(1),tmp_output:size(2),tmp_output:size(3))
		end
		self.output:select(1,i):copy(tmp_output)
	end
	return self.output
end


function SpatialConvolutionFromInput:updateGradInput(input,gradOutput)
	local data = input[1]
	local kernel = input[2]
	assert(kernel:size(2) == 1 and kernel:size(3) == 1, 'this needs to be set up to be more general')

	self.gradWeightView = self.gradWeightView or self.conv_module.gradWeight:view(self.nOutputPlane*self.nInputPlane,self.kW,self.kH)
	for i = 1,data:size(1) do
		if(i == 1) then
			self.gradData = self.gradData or data.new()
			self.gradKernel = self.gradKernel or data.new()
			self.gradData:resizeAs(data):zero()
			self.gradKernel:resizeAs(kernel):zero()
		end

		self.conv_module:zeroGradParameters()
		local ki = kernel:select(1,i):expandAs(self.conv_module.weight)
		self.conv_module.weight:copy(ki) 

		self.conv_module:forward(data:select(1,i))
		-- print('----')
		-- print('1')
		-- print(data:select(1,i):size(),data:select(1,i):type())
		-- 		print('2')

		-- print(gradOutput:select(1,i):size(),gradOutput:select(1,i):type())
		-- 		print('3')

		-- print(ki:size(),ki:type())
		-- 		print('4')

		-- print(self.conv_module.weight:type())
		-- 		print('5')

		-- print(self.conv_module.kW, self.conv_module.kH, self.conv_module.dW, self.conv_module.dH,self.conv_module.padW, self.conv_module.padH)
		-- print('')
		self.conv_module:backward(data:select(1,i),gradOutput:select(1,i))

		self.gradData:select(1,i):copy(self.conv_module.gradInput)
		
		if(kernel:size(2) == 1 and kernel:size(3) == 1) then
			self.gradKernel[i][1][1]:add(self.gradWeightView)
		else
			error('todo: set this up')
		end
	end

	self.gradInput = {self.gradData,self.gradKernel}

	return self.gradInput
end


