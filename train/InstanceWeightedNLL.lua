local InstanceWeightedNLL = torch.class('InstanceWeightedNLL')

function InstanceWeightedNLL:__init(soft_predictor,preprocess_ground_truth, preprocess_prediction, weight_func)
	self.soft_predictor = preprocess_prediction and nn.Sequential():add(soft_predictor):add(preprocess_prediction) or preprocess_prediction
	self.preprocess_ground_truth = preprocess_ground_truth
	self.weight_func = weight_func
	assert(weight_func)
end

function InstanceWeightedNLL:accumulate_gradient(x,y)
	local soft_pred = self.soft_predictor:forward(x)
	local yp = self.preprocess_ground_truth:forward(y)
	yp = yp:view(yp:size(1),1):long()
	assert(soft_pred:dim() == 2)

	local instance_weights = self.weight_func(x,y):view(yp:size())
	local scale = 1.0/instance_weights:sum()

	--TODO: inefficient
	if(Util:find_first_tensor(x):type() == "torch.CudaTensor") then
		yp = yp:typeAs(soft_pred)
	end
	local log_of_ground_truth = soft_pred:gather(2,yp)
	local loss = log_of_ground_truth:cmul(instance_weights):sum()*(-scale)

	self.d_pred = self.d_pred or soft_pred:clone()
	self.d_pred:zero()
	self.d_pred:scatter(2,yp,instance_weights)
	self.d_pred:mul(-scale)

	self.soft_predictor:backward(x,self.d_pred)

	return loss
end


function InstanceWeightedNLL:cuda()
	if(self.preprocess_ground_truth) then  self.preprocess_ground_truth:cuda() end
	self.soft_predictor:cuda()
end