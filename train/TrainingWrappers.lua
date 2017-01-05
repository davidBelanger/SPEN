local TrainingWrappers = torch.class('TrainingWrappers')


function TrainingWrappers:independent_training(network, criterion_name, prediction_shape, params)
	local return_all_iterates = params.return_all_iterates
	local num_iterates = params.num_iterates
	local training_net, criterion, preprocess_ground_truth, preprocess_prediction
	if(criterion_name == "ClassNLLCriterion") then
		training_net = nn.Sequential():add(network)
		criterion = nn.ClassNLLCriterion()
		local transform_to_log = nn.Sequential():add(nn.AddConstant(0.000001,true)):add(nn.Log())
		local all_but_domain_size = 1
		local domain_size = prediction_shape[#prediction_shape]
		for i = 1,(#prediction_shape - 1) do
			all_but_domain_size = all_but_domain_size*prediction_shape[i]
		end

		local iterate_weights
		--todo: implement
		-- if(params.TODO) then
		-- 	iterate_weights = TODO
		-- end
		if(not return_all_iterates) then
			preprocess_prediction   = nn.Sequential():add(transform_to_log):add(nn.View(all_but_domain_size,domain_size))
		else
			assert(num_iterates)
			training_net:add(nn.SelectTable(2))
			if(params.first_iter_to_apply_loss > 1) then
				training_net:add(nn.Narrow(1,params.first_iter_to_apply_loss,num_iterates - params.first_iter_to_apply_loss))
			end
			preprocess_prediction   = nn.Sequential():add(transform_to_log):add(nn.View(num_iterates,all_but_domain_size,domain_size)):add(nn.SplitTable(1))
			criterion = nn.RepeatedCriterion(criterion, iterate_weights)
		end
		preprocess_ground_truth = nn.View(all_but_domain_size)

	elseif(criterion_name == "MSECriterion") then
		assert(not return_all_iterates,'not implemented yet')

		training_net = network
		criterion = nn.MSECriterion()
		preprocess_ground_truth = nil
		preprocess_prediction = nil 
	end
	if(params.use_cuda) then
		training_net:cuda()
		criterion:cuda()
		if(preprocess_ground_truth) then preprocess_ground_truth:cuda() end
		if(preprocess_prediction) then preprocess_prediction:cuda() end
	end

	local prediction_penalty_net
	if(return_all_iterates and params.convergence_regularization_weight > 0) then
		prediction_penalty_net = TrainingWrappers:convergence_diagnostic_net(params)
		if(params.use_cuda) then prediction_penalty_net:cuda() end
	end
	local loss_wrapper = Independent(training_net, criterion, preprocess_ground_truth, preprocess_prediction, prediction_penalty_net)
	return loss_wrapper
end

function TrainingWrappers:ssvm_training(y_shape, model, criterion_name, gd_inference_TrainingWrappers, params)
	local loss_augmented_gd_TrainingWrappers = Util:copyTable(gd_inference_TrainingWrappers)
	loss_augmented_gd_TrainingWrappers.return_objective_value = true
	assert(not params.return_all_iterates, 'should not be penalizing all iterates if using ssvm loss')
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


function TrainingWrappers:convergence_diagnostic_net(params)
	--this takes in a sequence of iterates
	local num_iters = params.max_inference_iters
	local start = params.first_iter_to_penalize_convergence
	assert(start < num_iters)
	local len = num_iters - start

	local input = nn.Identity()()
	local iterates = input
	local left_iterates = nn.Narrow(1,start,len-1)(iterates)
	local right_iterates = nn.Narrow(1,start+1,len-1)(iterates)
	local diff = nn.Mean()(nn.View(-1)(nn.Square()(nn.CSubTable()({left_iterates,right_iterates}))))
	diff = nn.MulConstant(params.convergence_regularization_weight)(diff)
	return nn.gModule({input},{diff})
end

function TrainingWrappers:instance_weighted_training(network, criterion_name, prediction_shape, params)
	assert(criterion_name == "ClassNLLCriterion")
	local training_net = nn.Sequential():add(network)
	local return_all_iterates = params.return_all_iterates
	local num_iterates = params.num_iterates

	if(return_all_iterates) then 
		training_net:add(nn.SelectTable(2)) 
		if(params.first_iter_to_apply_loss > 1) then
			training_net:add(nn.Narrow(1,params.first_iter_to_apply_loss,num_iterates - params.first_iter_to_apply_loss))
		end
	end
	training_net:add(nn.AddConstant(0.000001,true)):add(nn.Log())

	local all_but_domain_size = 1
	local domain_size = prediction_shape[#prediction_shape]
	for i = 1,(#prediction_shape - 1) do
		all_but_domain_size = all_but_domain_size*prediction_shape[i]
	end
	local preprocess_ground_truth = nn.View(all_but_domain_size)
	local preprocess_prediction   = nn.Sequential():add(nn.View(all_but_domain_size,domain_size))

	if(return_all_iterates) then preprocess_prediction:add(nn.SplitTable(1)) end

	local weights
	local weight_func = function(x,y) 
		--TODO: this should be refactored, since much of this code is very specific to SRL

		local mask = x[2] --in this mask, if a cell is 1, it means that the corresponding arc was not filtered by the preprocessing

		weights = weights or mask:clone()
		weights:fill(params.negative_example_weight)

		weights:add(mask)

		if(params.false_positive_penalty ~= 0) then 
			--todo: this is assuming that the null_arc is index 1
			local mask2 = y:eq(1):typeAs(mask):cmul(mask) --in this mask, if a cell is 1, then it has a ground truth label of a null arc, but it wasn't filtered
			weights:add(params.false_positive_penalty,mask2)
		end

		if(return_all_iterates) then

			local full_size = {num_iterates}
			local reshape_size = {1}
			for i = 1,(#prediction_shape-1) do
				table.insert(full_size,prediction_shape[i])
				table.insert(reshape_size,prediction_shape[i])
			end
			weights:view(unpack(reshape_size)):expand(torch.LongStorage(full_size))
		end
		return weights
	end
	if(params.use_cuda) then
		training_net:cuda()
		preprocess_ground_truth:cuda()
		preprocess_prediction:cuda()
	end

	local prediction_penalty_net
	if(return_all_iterates and params.convergence_regularization_weight > 0) then
		prediction_penalty_net = TrainingWrappers:convergence_diagnostic_net(params)
		if(params.use_cuda) then prediction_penalty_net:cuda() end
	end
	local iterate_weights = nil --TODO
	local loss_wrapper = InstanceWeightedNLL(training_net, preprocess_ground_truth, preprocess_prediction, weight_func, return_all_iterates, iterate_weights, prediction_penalty_net)
	return loss_wrapper
end
