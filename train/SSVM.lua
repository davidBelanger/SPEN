local SSVM = torch.class('SSVM')

function SSVM:__init(full_energy_net,loss_augmented_predictor, preprocess_ground_truth, preprocess_prediction)
	assert(not preprocess_prediction, 'not implemented')
	self.preprocess_ground_truth = preprocess_ground_truth
	self.full_energy_net = full_energy_net
	self.loss_augmented_predictor = loss_augmented_predictor

end


function SSVM:accumulate_gradient(x,y)
	local yp = self.preprocess_ground_truth and self.preprocess_ground_truth:forward(y) or y
	local loss_augmented_prediction, loss_augmented_energy = unpack(self.loss_augmented_predictor:forward({yp,x}))
	print(yp:size())
	print(y:size())
	os.exit()
	local energy_on_ground_truth = self.full_energy_net:forward({yp,x})
	local hinge_loss_per_example = (energy_on_ground_truth - loss_augmented_energy):cmax(0) --todo: could pre-allocate this difference. 
	local loss_value = hinge_loss_per_example:mean()

	local hinge_grad = hinge_loss_per_example:gt(0):typeAs(hinge_loss_per_example)

	local standard_mode = true
	if(standard_mode) then
		self.full_energy_net:backward({yp,x},hinge_grad)
		hinge_grad:mul(-1)
		self.full_energy_net:forward({loss_augmented_prediction,x})
		self.full_energy_net:backward({loss_augmented_prediction,x},hinge_grad)
	else
		--this is a wacky idea where you explicitly penalize the predicted energy by backpropping through the gradient-based inference. it leads to divergence...
		hinge_grad:mul(-1)
		self.loss_augmented_predictor:forward({yp,x}) --todo: this could be avoided
		self.zero_grad = self.zero_grad or loss_augmented_prediction:clone()
		self.zero_grad:zero()
		local full_grad = {self.zero_grad,hinge_grad}
		self.loss_augmented_predictor:backward({yp,x},full_grad)

		hinge_grad:mul(-1)
		self.full_energy_net:forward({yp,x})
		self.full_energy_net:backward({yp,x},hinge_grad)

	end

	return loss_value
end


function SSVM:cuda()
	if(self.preprocess_ground_truth) then  self.preprocess_ground_truth:cuda() end
	self.full_energy_net:cuda()
	self.loss_augmented_predictor:cuda()
end