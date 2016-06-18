local TruncatedBackprop, Parent = torch.class('nn.TruncatedBackprop', 'nn.Sequential')

function TruncatedBackprop:__init()
	Parent.__init(self)
	self.doBackprop = true
end

function TruncatedBackprop:updateGradInput(input, gradOutput)
	if(self.doBackprop) then return Parent.updateGradInput(self,input,gradOutput) end
end

function TruncatedBackprop:accGradParameters(input, gradOutput,scale)
	if(self.doBackprop) then return Parent.accGradParameters(self,input,gradOutput,scale) end
end

function TruncatedBackprop:backward(input, gradOutput)
	if(self.doBackprop) then return Parent.backward(self,input,gradOutput) end
end

