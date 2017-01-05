local Independent = torch.class('Independent')

function Independent:__init(soft_predictor,criterion, preprocess_ground_truth, preprocess_prediction)
	self.soft_predictor = preprocess_prediction and nn.Sequential():add(soft_predictor):add(preprocess_prediction) or soft_predictor
	self.criterion = criterion
	self.preprocess_ground_truth = preprocess_ground_truth
end

function Independent:accumulate_gradient(x,y)
	local soft_pred = self.soft_predictor:forward(x)
	local yp = self.preprocess_ground_truth and self.preprocess_ground_truth:forward(y) or y
	local loss = self.criterion:forward(soft_pred,yp)
	local d_loss = self.criterion:backward(soft_pred,yp)
	self.soft_predictor:backward(x,d_loss)
	return loss
end


function Independent:cuda()
	if(self.preprocess_ground_truth) then  self.preprocess_ground_truth:cuda() end
	self.criterion:cuda()
	self.soft_predictor:cuda()
end