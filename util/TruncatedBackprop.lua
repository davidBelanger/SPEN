local TruncatedBackprop, Parent = torch.class('nn.TruncatedBackprop', 'nn.Container')

function TruncatedBackprop:__init(inner_module)
	assert(inner_module)
	self.inner_module = inner_module
	self.doBackprop = true
	self.modules = {}
	self:add(inner_module)

end

function TruncatedBackprop:setBackprop(value)
	self.doBackprop = value
end

function TruncatedBackprop:updateOutput(input)
	self.output = self.inner_module:forward(input)
	return self.output
end

function TruncatedBackprop:updateGradInput(input, gradOutput)
	if(self.doBackprop) then 
		self.gradInput = self.inner_module:updateGradInput(input,gradOutput)

		return self.gradInput
	else
		self.gradInput = self.gradInput or Util:deep_apply(input,function(t) return t:clone():zero() end)
		return self.gradInput
	end
end

function TruncatedBackprop:accGradParameters(input, gradOutput,scale)
	if(self.doBackprop) then 
		return self.inner_module:accGradParameters(input,gradOutput,scale)
	end
end

function TruncatedBackprop:backward(input, gradOutput)
	if(self.doBackprop) then 
		self.gradInput = self.inner_module:backward(input,gradOutput)
		return self.gradInput
	else
		self.gradInput = self.gradInput or Util:deep_apply(input,function(t) return t:clone():zero() end)
		return self.gradInput
	end
end

