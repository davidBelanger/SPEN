local RoundingPredictor = torch.class('RoundingPredictor')

function RoundingPredictor:__init(soft_predictor_net)
	self.soft_predictor_net = soft_predictor_net
end

function RoundingPredictor:predict(x)
	local ysoft = self.soft_predictor_net:forward(x)

	local values, inds = ysoft:max(ysoft:nDimension())
	local ndims = inds:nDimension()
	local pred = inds:select(ndims,1)
	return pred
end

function RoundingPredictor:cuda()
	self.soft_predictor_net:cuda()
end

function RoundingPredictor:evaluate()
	self.soft_predictor_net:evaluate()
end

function RoundingPredictor:training()
	self.soft_predictor_net:training()
end