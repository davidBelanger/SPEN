local Predicate, parent = torch.class("nn.Predicate", "nn.Module")

function Predicate:__init(func)
	self.func = func
	self.initialized = false
	self.output = torch.ByteTensor(1)

end

--todo: this could be extended to have multidimensional byttensor outputs

--todo: this could easily be extended to make more inputs as arguments

function Predicate:updateOutput(input)
	local value = self.func(unpack(input))
	if(value) then
		self.output:fill(1)
	else
		self.output:fill(0)
	end
	return self.output
end

function Predicate:updateGradInput(input, gradOutput)
	if(not self.initialized) then
		self.gradInput = {}
   		table.insert(self.gradInput,input[1]:clone():zero())
   		table.insert(self.gradInput,input[2]:clone():zero())
   		self.initialized = true
	end

   return self.gradInput
end