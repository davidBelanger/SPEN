require 'PredictionTask'
local LinkPrediction, parent = torch.class('LinkPrediction','PredictionTask')

function LinkPrediction:__init(config)
	parent.__init(self,config)
	self.config = config
end



function LinkPrediction:evaluate_batch(predictor,batch)
	local y, x = unpack(batch)
	y = y:squeeze()
	local y_pred = predictor(x)
	return y_pred:eq(y):float():sum()/y:nElement()
end


