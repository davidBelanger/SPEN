local seed = 12345
torch.manualSeed(seed)

require 'Imports'

local cmd = GeneralOptions:get_flags()
local params = cmd:parse(arg)

if(params.profile == 1) then
   require 'Pepperfish'  
   profiler = new_profiler()
   profiler:start()
end

print(params)

local use_cuda = params.gpuid >= 0
params.use_cuda= use_cuda
if(use_cuda)then
    print('USING GPU '..params.gpuid)
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpuid + 1) 
    cutorch.manualSeed(seed)
    if(params.cudnn == 1) then
    	print('using cudnn')
		require 'cudnn'
	end
end


local load_problem
if(params.problem == "SequenceTagging") then
	load_problem = function(params)
		local problem_config = torch.load(params.problem_config)
		problem_config.batch_size = params.batch_size

		local y_shape = {problem_config.batch_size,problem_config.length,problem_config.domain_size} 
		problem_config.y_shape = y_shape
		local model = ChainSPEN(problem_config,params)
		local evaluator_factory =  function(batcher, soft_predictor)
			local hard_predictor = RoundingPredictor(soft_predictor,y_shape)
			return HammingEvaluator(batcher, function(x) return hard_predictor:predict(x) end)
		end 
		local preprocess_func = nil
		local train_batcher = BatcherFromFile(params.train_list, preprocess_func, params.batch_size, use_cuda)
		local test_batcher  = BatcherFromFile(params.test_list, preprocess_func, params.batch_size, use_cuda)
		return model, y_shape, evaluator_factory, preprocess_func, train_batcher, test_batcher
	end
elseif(params.problem == "MultiLabelClassification") then
	load_problem = function(params)
		local problem_config = torch.load(params.problem_config)
		problem_config.batch_size = params.batch_size

		local y_shape = {problem_config.batch_size,problem_config.label_dim,2} 
		problem_config.y_shape = y_shape
		local model = MLCSPEN(problem_config,params)
		local evaluator_factory =  function(batcher, soft_predictor)
			return MultiLabelEvaluation(batcher, soft_predictor, problem_config.prediction_thresh, params.results_file)
		end 

		local adder = nn.AddConstant(1)
		if(use_cuda) then adder:cuda() end

		preprocess_func = function(a,b,c) return adder:forward(a):clone(), b, c end --TODO: make this unnecessary by preprocessing data differently. Right now, the labels are 0-indexed, so we have to add one.
		
		local train_batcher = BatcherFromFile(params.train_list, preprocess_func, params.batch_size, use_cuda)
		local test_batcher  = BatcherFromFile(params.test_list, preprocess_func, params.batch_size, use_cuda)
		return model, y_shape, evaluator_factory, preprocess_func, train_batcher, test_batcher
	end
elseif(params.problem == "Depth") then
	load_problem = function(params)
		local problem_config = torch.load(params.problem_config)
		problem_config.batch_size = params.batch_size
		
		local y_shape, train_preprocess_func 
		if(problem_config.use_random_crops == 1) then
			local crop_height = 96 --these are the crop sizes used in the proximalnet paper. TODO: surface command line options for these
			local crop_width = 128
			y_shape = {problem_config.batch_size,crop_height,crop_width}
			local a_crop_contiguous, b_crop_contiguous
			local y_crop_start_max = problem_config.height - crop_height
			local x_crop_start_max = problem_config.width  - crop_width

			--This randomly crops the images in order to speed up training
			--Note that it uses the same crop locations for every image in the minibatch
			train_preprocess_func = function(a,b,c)	
				local y_start = torch.rand(1):mul(y_crop_start_max):ceil()[1]
				local x_start = torch.rand(1):mul(x_crop_start_max):ceil()[1]
				local a_crop = a:narrow(2,y_start,crop_height):narrow(3,x_start,crop_width)
				local b_crop = b:narrow(2,y_start,crop_height):narrow(3,x_start,crop_width)
				a_crop_contiguous = a_crop_contiguous or a_crop:clone()
				a_crop_contiguous:copy(a_crop)

				b_crop_contiguous = b_crop_contiguous or b_crop:clone()
				b_crop_contiguous:copy(b_crop)

				return a_crop_contiguous, b_crop_contiguous, c
			end
		else
			y_shape = {problem_config.batch_size,problem_config.height,problem_config.width} 
		end

		problem_config.y_shape = y_shape
		local model = DepthSPEN(problem_config,params)
		local evaluator_factory =  function(batcher, soft_predictor)
			return PSNREvaluator(batcher, function(x) return soft_predictor:forward(x) end)
		end 


		local train_batcher = BatcherFromFile(params.train_list, train_preprocess_func, params.batch_size, use_cuda)

		--NOTE: this doesn't return the actual test set score, but an approximation using random crops on the dev set.
		--It will require some more engineering to be able to actually run on the full-size test images, as the network expects smaller images.
		local test_batcher  = BatcherFromFile(params.test_list, train_preprocess_func, params.batch_size, use_cuda)
		return model, y_shape, evaluator_factory, preprocess_func, train_batcher, test_batcher
	end
elseif(params.problem == 'SRL') then
	load_problem = function(params)
		local problem_config = torch.load(params.problem_config)
		
		-- local train_labels_file = "srl/processed_data/conll2005/train.arcs.torch.small"
		-- local train_features_file = "srl/processed_data/conll2005/train.features.torch.small"

		-- local test_labels_file = "srl/processed_data/conll2005/train.arcs.torch.small"
		-- local test_features_file = "srl/processed_data/conll2005/train.features.torch.small"

		local train_labels_file = "srl/processed_data/conll2005_3/train.arcs.torch"
		local train_features_file = "srl/processed_data/conll2005_3/train.all.features.torch"
		local train_collision_file_base = "srl/processed_data/conll2005_3/train.collisions"
		local srl_labels_map_file = "srl/processed_data/conll2005_3/role_to_id.txt"

		local test_labels_file = "srl/processed_data/conll2005_3/dev.arcs.torch"
		local test_features_file = "srl/processed_data/conll2005_3/dev.all.features.torch"
		local test_collision_file_base = "srl/processed_data/conll2005_3/dev.collisions"
		
		local domain_size = problem_config.domain_size
		local null_arc_index = problem_config.null_arc_index
		local batch_size = params.batch_size
		local feature_dim = problem_config.feature_dim
		problem_config.feature_dim = problem_config.feature_dim + 1 -- add one because the batcher adds one more feature 

		problem_config.load_node_features  = true --todo: surface
		local train_node_files
		local test_node_files
		if(problem_config.load_node_features) then
			train_node_files = {"srl/processed_data/conll2005_3/train.predicate.features.torch","srl/processed_data/conll2005_3/train.arg.features.torch"}
			test_node_files = {"srl/processed_data/conll2005_3/dev.predicate.features.torch","srl/processed_data/conll2005_3/dev.arg.features.torch"}
		end

		local train_batcher = SRLBatcher(train_labels_file,train_features_file,train_collision_file_base, train_node_files, batch_size, feature_dim, problem_config.max_predicates, problem_config.max_arguments, null_arc_index, params.use_cuda, true, params.shuffle==1)
		
		problem_config.max_predicates = train_batcher.max_rows --todo: this isn't totally correct. It assumes that the biggest example in the training data is bigger than the biggest example in the test data
		problem_config.max_arguments = train_batcher.max_cols

		local test_batcher =  SRLBatcher(test_labels_file,test_features_file,test_collision_file_base, test_node_files, batch_size, feature_dim, problem_config.max_predicates, problem_config.max_arguments, null_arc_index, params.use_cuda, false, false)


		local y_shape = {batch_size,problem_config.max_predicates,problem_config.max_arguments,domain_size}

		problem_config.y_shape = y_shape
		local model = SRLSPEN(problem_config, params)

		local preprocess_func = nil
		local evaluator_factory = function(batcher, soft_predictor)
			local hard_predictor = RoundingPredictor(soft_predictor,y_shape)
			local prediction_writing_info
			if(problem_config.use_official_scoring_script == 1) then
				prediction_writing_info = {
					labels = test_batcher.labels,
					out_file_base = params.out_dir.."/test_pred",
					label_map_file = srl_labels_map_file,
					evaluation_script = "srl/scripts/evaluate_dev.sh"
				}
			end
			return SRLEvaluator(batcher, hard_predictor, problem_config.null_arc_index, prediction_writing_info)
		end  

		return model, y_shape, evaluator_factory, preprocess_func, train_batcher, test_batcher
	end

else
	error('invalid problem type')
end

local model, y_shape, evaluator_factory, preprocess_func, train_batcher, test_batcher = load_problem(params)

local pretrain_train_config = {}
do
	pretrain_train_config.soft_predictor = model.classifier_network
	pretrain_train_config.modules_to_update = model.classifier_network
	pretrain_train_config.stop_feature_backprop = false
	pretrain_train_config.stop_unary_backprop = false

	local criterion_name = (params.continuous_outputs == 1) and "MSECriterion" or "ClassNLLCriterion"
	print(params)
	if(params.instance_weighted_loss == 1) then
		pretrain_train_config.loss_wrapper = TrainingWrappers:instance_weighted_training(model.classifier_network, criterion_name, y_shape, params)
	else
		pretrain_train_config.loss_wrapper = TrainingWrappers:independent_training(model.classifier_network, criterion_name, y_shape, params)
	end
	pretrain_train_config.items_to_save = {
		classifier = model.classifier_network
	}
end

local full_train_config = {}
do 
	params.return_all_iterates = params.penalize_all_iterates == 1
	params.num_iterates = params.max_inference_iters
	local gd_inference_config = GradientBasedInferenceConfig:get_gd_inference_config(params)
	local full_gd_prediction_net = GradientBasedInference(y_shape, gd_inference_config):spen_inference(model)
	gd_prediction_net = full_gd_prediction_net
	if(params.return_all_iterates) then
		gd_prediction_net = nn.Sequential():add(full_gd_prediction_net):add(nn.SelectTable(1))
	end
	--initialize the unaries from a loaded model
	if(params.init_classifier ~= "") then
		assert(not (params.init_full_net ~= ""), "shouldn't be initializing both classifier and full energy network from file")
		print('initializing classifier from '..params.init_classifier)
		model.classifier_network:getParameters():copy(torch.load(params.init_classifier):getParameters())
	end
	if(params.init_full_net ~= "") then
		print('initializing parameters from '..params.init_full_net)
		if(params.use_cuda) then gd_prediction_net:double() end --the fact that we have to do this is mysterious. 
		gd_prediction_net:getParameters():copy(torch.load(params.init_full_net):getParameters())
		if(params.use_cuda) then gd_prediction_net:cuda() end
	end

	full_train_config.soft_predictor = gd_prediction_net
	full_train_config.modules_to_update = gd_prediction_net

	if(params.training_method == "E2E") then
		local criterion_name = (params.continuous_outputs == 1) and "MSECriterion" or "ClassNLLCriterion"
		if(params.instance_weighted_loss == 1) then
			full_train_config.loss_wrapper = TrainingWrappers:instance_weighted_training(full_gd_prediction_net, criterion_name, y_shape, params)
		else
			full_train_config.loss_wrapper = TrainingWrappers:independent_training(full_gd_prediction_net, criterion_name, y_shape, params)
		end
	elseif(params.training_method == "SSVM") then
		local criterion_name = 'MSECriterion' --todo: surface an option for this
		assert(not params.return_all_iterates)
		full_train_config.loss_wrapper = TrainingWrappers:ssvm_training(y_shape, model, criterion_name, gd_inference_config, params)
	else
	    error('invalid training method')
	end

	full_train_config.items_to_save = {
		predictor = gd_prediction_net,
		energy_net = model:full_energy_net()
	}
end

local clamp_features_train_config = Util:copyTable(full_train_config) --this is a copy by reference
clamp_features_train_config.stop_feature_backprop = true

local clamp_unaries_train_config = Util:copyTable(full_train_config) --this is a copy by reference
clamp_unaries_train_config.stop_unary_backprop = true
clamp_unaries_train_config.stop_feature_backprop = true


local train_configurations = {
	pretrain = pretrain_train_config,
	clamp_features = clamp_features_train_config,
	clamp_unaries = clamp_unaries_train_config,
	full = full_train_config
}


local function evaluate_only(config,params)
	if(params.use_cuda) then config.soft_predictor:cuda() end
	local evaluator = evaluator_factory(test_batcher, config.soft_predictor)
	evaluator:evaluate(0)
end

local function train(config, params, name)
	assert(config)

	if(params.use_cuda) then
		config.loss_wrapper:cuda()
		config.soft_predictor:cuda()
	end
	model:set_feature_backprop( not config.stop_feature_backprop )
	model:set_unary_backprop( not config.stop_unary_backprop )

	local callbacks = {}
	local evaluator = evaluator_factory(test_batcher, config.soft_predictor)
	local evaluate = Callback(function(data) return evaluator:evaluate(data.epoch) end, params.evaluation_frequency)
	table.insert(callbacks,evaluate)


	local opt_state = {}
	if(params.init_opt_state ~= "") then 
		print('loading opt_state from '..params.init_opt_state)
		opt_state = torch.load(params.init_opt_state)
	end
	local optimization_config = {
		opt_state = opt_state,
		opt_config = {
					  learningRate=params.learning_rate,
					  learningRateDecay=0, --this gets updated by lr_start below
					  beta1 = params.adam_beta1,
					  beta2 = params.adam_beta2,
					  epsilon=params.adam_epsilon,
					  weightDecay=params.l2
					 },
		opt_method = optim.adam,
		gradient_clip = params.gradient_clip,
		regularization = config.regularization,
		modules_to_update = config.modules_to_update
	}

	local general_config = {
		num_epochs = params.num_epochs,
		batches_per_epoch = params.batches_per_epoch,
		batch_size = params.batch_size,
		assert_nan = true,
	}

	local nonzero_learning_rate_set = false
	local function set_lr(data)
		if(data.epoch > params.learning_rate_decay_start and not set_nonzero_learning_rate) then
			optimization_config.opt_config.learningRateDecay = params.learning_rate_decay
			nonzero_learning_rate_set = true
		end
	end
	local lr_start = Callback(set_lr, 1)
	table.insert(callbacks,lr_start)

	if(params.icnn == 1) then
		local params_to_clamp = model.global_potentials_network:parameters()
		--todo: we actually don't need to clamp the biases
		local function clamp()
			Util:deep_apply(params_to_clamp,function(t) t:cmax(0) end)
		end
		general_config.post_process_parameter_update = clamp
	end

	config.items_to_save.opt_state = optimization_config.opt_state
	local saved_model_base = params.model_file.."-"..name
	local saver = Saver(saved_model_base,config.items_to_save)
	local save = Callback(function(data) return saver:save(data.epoch) end, params.save_frequency)
	table.insert(callbacks,save)

	Train(config.loss_wrapper,train_batcher, optimization_config, general_config, callbacks):train()
end

if(params.evaluate_classifier_only == 1) then
	evaluate_only(train_configurations.pretrain, params)
	os.exit()
end
if(params.evaluate_spen_only == 1) then
	evaluate_only(train_configurations.full, params)
	os.exit()
end

for params_file in io.lines(params.training_configs) do
	print('loading specific training config from '..params_file)
	local specific_params = torch.load(params_file)

	local mode = specific_params.training_mode
	if(specific_params.num_epochs > 0) then 
		print(specific_params)

		local all_params = Util:copyTable(params)
		for k,v in pairs(specific_params) do 
			assert(not all_params[k],'repeated key: '..k)
			all_params[k] = v 
		end

		print('starting training for mode: '..mode)
		if(mode == "pretrain_unaries") then
			train(train_configurations.pretrain, all_params, mode)
		elseif(mode == "clamp_unaries") then
			train(train_configurations.clamp_unaries, all_params, mode)
		elseif(mode == "clamp_features") then
			train(train_configurations.clamp_features, all_params, mode)
		elseif(mode == "update_all") then
			train(train_configurations.full, all_params, mode)
		else
			error('invalid training mode: '..mode)
		end
	end
end


if(params.profile == 1) then
	profiler:stop()
	local report = "profile.txt"
	print('writing profiling report to: '..report)
    local outfile = io.open( report, "w+" )
    profiler:report( outfile )
    outfile:close()
end



