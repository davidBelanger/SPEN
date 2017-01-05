local Callback = torch.class('Callback')

function Callback:__init(lambda, frequency)
	self.lambda = lambda
	self.frequency = frequency
end

function Callback:run(data)
	if(data.epoch % self.frequency == 0) then
		self.lambda(data)
	end
end