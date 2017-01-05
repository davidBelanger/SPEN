local TrainingWrappers = torch.class('TrainingWrappers')


function TrainingWrappers:independent_training(network, criterion_name, prediction_shape, params)
	assert(criterion_name == "ClassNLLCriterion")
	local training_net, criterion, preprocess_ground_truth, preprocess_prediction
	if(criterion_name == "ClassNLLCriterion") then
		training_net = nn.Sequential():add(network):add(nn.AddConstant(0.000001,true)):add(nn.Log())
		criterion = nn.ClassNLLCriterion()

		local all_but_domain_size = 1
		local domain_size = prediction_shape[#prediction_shape]
		for i = 1,(#prediction_shape - 1) do
			all_but_domain_size = all_but_domain_size*prediction_shape[i]
		end
		preprocess_ground_truth = nn.View(all_but_domain_size)
		preprocess_prediction   = nn.View(all_but_domain_size,domain_size)
	end
	if(params.use_cuda) then
		training_net:cuda()
		criterion:cuda()
		preprocess_ground_truth:cuda()
		preprocess_prediction:cuda()
	end
	local loss_wrapper = Independent(training_net, criterion, preprocess_ground_truth, preprocess_prediction)
	return loss_wrapper
end

function TrainingWrappers:ssvm_training(y_shape, model, criterion_name, gd_inference_TrainingWrappers, params)
	local loss_augmented_gd_TrainingWrappers = Util:copyTable(gd_inference_TrainingWrappers)
	loss_augmented_gd_TrainingWrappers.return_objective_value = true
	
	assert(criterion_name == 'MSECriterion')
	local cost_net
	if(criterion_name == 'MSECriterion') then
		cost_net = SquaredLossPerBatchItem(#y_shape)
	end
    local loss_augmented_predictor = GradientBasedInference(y_shape, loss_augmented_gd_TrainingWrappers):loss_augmented_spen_inference(model,cost_net)
    local full_energy_net = model:full_energy_net()
    local domain_size = y_shape[#y_shape]
    local preprocess_ground_truth = nn.OneHot(domain_size) 
    local preprocess_prediction = nil
    if(params.use_cuda) then
		training_net:cuda()
		criterion:cuda()
		preprocess_ground_truth:cuda()
	end
    loss_wrapper = SSVM(full_energy_net,loss_augmented_predictor,  preprocess_ground_truth, preprocess_prediction)
	return loss_wrapper
end


function TrainingWrappers:instance_weighted_training(network, criterion_name, prediction_shape, params)
	assert(criterion_name == "ClassNLLCriterion")
	local training_net = nn.Sequential():add(network):add(nn.AddConstant(0.000001,true)):add(nn.Log())

	local all_but_domain_size = 1
	local domain_size = prediction_shape[#prediction_shape]
	for i = 1,(#prediction_shape - 1) do
		all_but_domain_size = all_but_domain_size*prediction_shape[i]
	end
	local preprocess_ground_truth = nn.View(all_but_domain_size)
	local preprocess_prediction   = nn.View(all_but_domain_size,domain_size)

	local weights
	local weight_func = function(x,y) 
		local mask = x[2] 
		weights = weights or mask:clone()
		weights:fill(params.negative_example_weight)

		weights:add(mask)
		return weights
	end
	if(params.use_cuda) then
		training_net:cuda()
		preprocess_ground_truth:cuda()
		preprocess_prediction:cuda()
	end
	local loss_wrapper = InstanceWeightedNLL(training_net, preprocess_ground_truth, preprocess_prediction, weight_func)
	return loss_wrapper
end
