require 'PredictionTask'
local SequenceTagging, parent = torch.class('SequenceTagging','PredictionTask')

function SequenceTagging:__init(config)
	parent.__init(self,config)
	self.config = config
end



function SequenceTagging:evaluate_batch(predictor,batch)
	local y, x = unpack(batch)
	y = y:squeeze()
	local y_pred = predictor(x)
	return y_pred:eq(y):float():sum()/y:nElement()
end


