local Independent = torch.class('Independent')

function Independent:__init(soft_predictor,criterion, preprocess_ground_truth, preprocess_prediction, prediction_penalty_net)
	self.soft_predictor = soft_predictor
	self.preprocess_prediction = preprocess_prediction or nn.Identity()
	self.criterion = criterion
	self.preprocess_ground_truth = preprocess_ground_truth or nn.Identity()
	self.prediction_penalty_net = prediction_penalty_net
end

function Independent:accumulate_gradient(x,y)
	local raw_pred = self.soft_predictor:forward(x)
	local soft_pred = self.preprocess_prediction:forward(raw_pred)
	local yp = self.preprocess_ground_truth and self.preprocess_ground_truth:forward(y) or y
	local loss = self.criterion:forward(soft_pred,yp)
	local d_loss = self.criterion:backward(soft_pred,yp)
	local d_raw = self.preprocess_prediction:backward(raw_pred,d_loss)

	if(self.prediction_penalty_net) then
		loss = loss + self.prediction_penalty_net:forward(raw_pred)[1]
		assert(self.prediction_penalty_net.output:nElement() == 1)
		self.constant_ones = self.constant_ones or torch.ones(1):typeAs(yp)

		local d_penalty = self.prediction_penalty_net:backward(raw_pred,self.constant_ones)
		Util:deep_apply_inplace_two_arg(d_raw,d_penalty,function(t,s) return t:add(s) end) 
	end

	self.soft_predictor:backward(x,d_raw)

	return loss
end


function Independent:cuda()
	if(self.preprocess_ground_truth) then  self.preprocess_ground_truth:cuda() end
	self.criterion:cuda()
	self.soft_predictor:cuda()
end