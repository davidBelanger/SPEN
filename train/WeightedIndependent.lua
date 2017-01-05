local InstanceWeightedBCE = torch.class('InstanceWeightedBCE')

function InstanceWeightedBCE:__init(soft_predictor,preprocess_ground_truth, preprocess_prediction, weight_func)
	self.soft_predictor = preprocess_prediction and nn.Sequential():add(soft_predictor):add(preprocess_prediction) or preprocess_prediction
	self.preprocess_ground_truth = preprocess_ground_truth
	self.weight_func = weight_func
end

function InstanceWeightedBCE:accumulate_gradient(x,y)
	local soft_pred = self.soft_predictor:forward(x)
	local yp = self.preprocess_ground_truth:forward(y)
	yp = yp:view(yp:size(1),1)
	assert(soft_pred:dim() == 2)

	assert(yp:dim() == 1)
	assert(soft_pred:dim() == 2)

	local instance_weights = self.weight_func(x,y):view(yp:size())
	local log_of_ground_truth = soft_pred:gather(2,yp)
	local loss = log_of_ground_truth:cmul(instance_weights):sum()*-1

	self.d_pred = self.d_pred or soft_pred:clone()
	self.d_pred:zero()
	instance_weights = instance_weights:view(yp:size())
	self.d_pred:scatter(2,yp,instance_weights)
	self.d_pred:mul(-1)

	self.soft_predictor:backward(x,d_pred)
	return loss
end


function InstanceWeightedBCE:cuda()
	if(self.preprocess_ground_truth) then  self.preprocess_ground_truth:cuda() end
	self.soft_predictor:cuda()
end