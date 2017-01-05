local Cond, parent = torch.class('nn.Cond', 'nn.Container')


function Cond:__init(network_if_true, network_if_false)
	self.network_if_true  = network_if_true
	self.network_if_false = network_if_false
	self.modules = {}
	self:add(network_if_true)
	self:add(network_if_false)
end


--the boolean_condition should be a size 1 ByteTensor
--input is assumed to be {boolean_condition, input_to_networks}
function Cond:updateOutput(input)
	local condition = input[1]
	--assert(condition:type() == "torch.ByteTensor", "received: "..condition:type()) --this only works on the cpu
	if(condition[1] == 1) then
		self.output = self.network_if_true:forward(input[2])
	else
		self.output = self.network_if_false:forward(input[2])
	end
	return self.output
end


function Cond:updateGradInput(input,gradOutput)
	local condition = input[1]
	self.cond_grad = self.cond_grad or torch.Tensor(1):typeAs(input[1]):fill(0)
	if(condition[1] == 1) then
		self.gradInput = {self.cond_grad,self.network_if_true:updateGradInput(input[2],gradOutput)}
	else
		self.gradInput = {self.cond_grad,self.network_if_false:updateGradInput(input[2],gradOutput)}
	end
	return self.gradInput
end

function Cond:accGradParameters(input,gradOutput,scale)
	local condition = input[1]
	if(condition[1] == 1) then
		self.network_if_true:accGradParameters(input[2],gradOutput,scale)
	else
		self.network_if_false:accGradParameters(input[2],gradOutput,scale)
	end
end