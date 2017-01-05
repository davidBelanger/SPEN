local InstanceWeightedNLL = torch.class('InstanceWeightedNLL')

function InstanceWeightedNLL:__init(soft_predictor,preprocess_ground_truth, preprocess_prediction, weight_func, tiled_predictions, iterate_weights,prediction_penalty_net)
	self.soft_predictor = soft_predictor
	self.preprocess_prediction = preprocess_prediction
	self.preprocess_ground_truth = preprocess_ground_truth
	self.weight_func = weight_func
	self.tiled_predictions = tiled_predictions
	self.iterate_weights = iterate_weights
	assert(weight_func)
	self.prediction_penalty_net = prediction_penalty_net
end

function InstanceWeightedNLL:accumulate_gradient(x,y)
	local raw_pred = self.soft_predictor:forward(x)
	local full_soft_pred = self.preprocess_prediction:forward(raw_pred)
	local yp = self.preprocess_ground_truth:forward(y)
	yp = yp:view(yp:size(1),1):long()

	local instance_weights = self.weight_func(x,y):view(yp:size())
	local scale = 1.0/instance_weights:sum()

	--TODO: inefficient
	if(Util:find_first_tensor(x):type() == "torch.CudaTensor") then
		yp = yp:cuda()
	end

	local loss
	if(not self.tiled_predictions) then
		local log_of_ground_truth = full_soft_pred:gather(2,yp)
		loss = log_of_ground_truth:cmul(instance_weights):sum()*(-scale)

		self.d_pred = self.d_pred or full_soft_pred:clone()
		self.d_pred:zero()
		self.d_pred:scatter(2,yp,instance_weights)
		self.d_pred:mul(-scale)
	else
		self.d_pred = self.d_pred or Util:deep_clone(full_soft_pred)
		Util:deep_apply_inplace(self.d_pred,function(t) return t:zero() end)
		loss = 0

		for i = 1,#full_soft_pred do
			local soft_pred = full_soft_pred[i]
			local log_of_ground_truth = soft_pred:gather(2,yp)
			local weight = self.iterate_weights and self.iterate_weights[i] or 1.0/#full_soft_pred

			loss = loss + weight*log_of_ground_truth:cmul(instance_weights):sum()*(-scale)


			self.d_pred[i]:scatter(2,yp,instance_weights)
			self.d_pred[i]:mul(-scale*weight)
		end
	end
	local d_raw = self.preprocess_prediction:backward(raw_pred,self.d_pred)
	
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


function InstanceWeightedNLL:cuda()
	if(self.preprocess_ground_truth) then  self.preprocess_ground_truth:cuda() end
	self.soft_predictor:cuda()
end