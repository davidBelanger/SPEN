local FFT, parent = torch.class('nn.FFT', 'nn.Module')
local signal = require 'signal'
function FFT:__init()
	parent.__init(self)
end


function FFT:updateOutput(input)
	local mb = input:size(1)
	local s = input:size(2)
	self.output:resize(mb,s,s,2)
	self.tmp_input = self.tmp_input or input:clone():float()
	self.tmp_input:resize(input:size()):copy(input)

	for i = 1,mb do
		self.output[i]:copy(signal.fft2(self.tmp_input[i]))
	end
	return self.output
end

function FFT:updateGradInput(input,gradOutput)
	self.gradInput:resizeAs(input)
	local mb = input:size(1)
	local s = input:size(2)
	
	self.tmp_gradoutput = self.tmp_gradoutput or input:clone():float()
	self.tmp_gradoutput:resize(gradOutput:size()):copy(gradOutput)
	

	for i = 1,mb do
		local iff = signal.ifft2(self.tmp_gradoutput[i])
		self.gradInput[i]:copy(iff:select(3,1))
	end
	return self.gradInput
end


