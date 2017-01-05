local Evaluator = torch.class('Evaluator')

--todo: this should take a factory for an iterator

function Evaluator:__init(batcher,predict_func)
	self.batcher = batcher
	self.predict_func = predict_func
end

function Evaluator:evaluate_batch(batch)
	local y = batch[1]
	local x = batch[2]
	local y_pred = self.predict_func(x)
	local num = y:nElement() --todo: use the data returned by the batcher, which may have num_actual_data
	return self:score_prediction(y_pred,y), num
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