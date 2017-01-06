local Evaluator = torch.class('Evaluator')


--this is an abstract class that requires an implementation of self:score_prediction(y_pred,y,num). It returns the total loss for a batch (not averaged by the size of the batch)
function Evaluator:__init(batcher,predict_func)
	self.batcher = batcher
	self.predict_func = predict_func
end

function Evaluator:evaluate_batch(batch)
	local y = batch[1]
	local x = batch[2]
	local num = batch[3]
	local y_pred = self.predict_func(x)
	y_pred = y_pred:narrow(1,1,num)
	y = y:narrow(1,1,num)
	return self:score_prediction(y_pred,y, num), num
end

function Evaluator:evaluate(msg)
	local sum = 0
	local iterator = self.batcher:get_iterator()
	local batch = iterator()
	local count = 1
	while(batch) do
		local score, num = self:evaluate_batch(batch)
		sum = sum + score
		count = count + num
		batch = iterator()
	end
	local acc = sum*1.0/count
	msg = msg or ""
	print(msg.." Accuracy = "..acc)
	return acc
end