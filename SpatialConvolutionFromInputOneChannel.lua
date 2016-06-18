local SpatialConvolutionFromInputOneChannel, parent = torch.class('nn.SpatialConvolutionFromInputOneChannel', 'nn.Container')


--NOTE: this assumes that the convolution is done with zero bias

function SpatialConvolutionFromInputOneChannel:__init(kW, kH, dW, dH, padW, padH)
	parent.__init(self)
	self.dW = dW or 1
   	self.dH = dH or 1
   	self.padW = padW or 0
   	self.padH = padH or 0
	self.kW = kW
	self.kH = kH

	table.insert(self.modules,self.conv_module)
end

function SpatialConvolutionFromInputOneChannel:parameters()
	return nil
end

function SpatialConvolutionFromInputOneChannel:setConvModule(minibatchsize)
	if(minibatchsize ~= self.minibatchsize) then
		self.minibatchsize = minibatchsize
		print(1,minibatchsize, self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
		self.conv_module = nn.SpatialConvolution(1,minibatchsize, self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
		self.conv_module.bias:zero()
		self.weight = nil 
		self.bias = nil
	end
end

function SpatialConvolutionFromInputOneChannel:updateOutput(input)

	assert(#input == 2)
	local data = input[1]
	local kernel = input[2]
	local minibatchsize = data:size(1)
	self:setConvModule(minibatchsize)
	self.conv_module.weight:copy(kernel:view(minibatchsize,1,self.kW,self.kH))

	self.output = self.conv_module:forward(data)
	return self.output
end


function SpatialConvolutionFromInputOneChannel:updateGradInput(input,gradOutput)
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
		self.conv_module.weight:copy(kernel:select(1,i):expandAs(self.conv_module.weight)) 

		self.conv_module:forward(data:select(1,i))
		self.conv_module:backward(data:select(1,i),gradOutput:select(1,i))

		self.gradData:select(1,i):copy(self.conv_module.gradInput)
		
		if(kernel:size(2) == 1 and kernel:size(3) == 1) then
			if(not self.tmpSum) then 
				self.tmpSum = torch.sum(self.gradWeightView,1)
			else
				torch.sum(self.tmpSum,self.gradWeightView,1)
			end
			self.gradKernel[i][1][1]:add(1/(self.nInputPlane*self.nOutputPlane),self.tmpSum:squeeze())
		else
			error('todo: set this up')
		end
	end

	self.gradInput = {self.gradData,self.gradKernel}

	return self.gradInput
end


