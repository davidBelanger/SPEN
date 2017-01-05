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
    if(params.cudnn) then
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
		preprocess_func = function(a,b,c) return adder:forward(a):clone(), b, c end
		
		local train_batcher = BatcherFromFile(params.train_list, preprocess_func, params.batch_size, use_cuda)
		local test_batcher  = BatcherFromFile(params.test_list, preprocess_func, params.batch_size, use_cuda)
		return model, y_shape, evaluator_factory, preprocess_func, train_batcher, test_batcher
	end
elseif(params.problem == 'SRL') then
	load_problem = function(params)
		local problem_config = torch.load(params.problem_config)
		
		-- local train_labels_file = "srl/processed_data/conll2005/train.arcs.torch.small"
		-- local train_features_file = "srl/processed_data/conll2005/train.features.torch.small"

		-- local test_labels_file = "srl/processed_data/conll2005/train.arcs.torch.small"
		-- local test_features_file = "srl/processed_data/conll2005/train.features.torch.small"

		local train_labels_file = "srl/processed_data/conll2005/train.arcs.torch"
		local train_features_file = "srl/processed_data/conll2005/train.features.torch"
		local train_collision_file_base = "srl/processed_data/conll2005/train.collisions"

		local test_labels_file = "srl/processed_data/conll2005/dev.arcs.torch"
		local test_features_file = "srl/processed_data/conll2005/dev.features.torch"
		local test_collision_file_base = "srl/processed_data/conll2005/dev.collisions"
		
		local domain_size = problem_config.domain_size
		local null_arc_index = problem_config.null_arc_index
		local batch_size = params.batch_size
		local feature_dim = problem_config.feature_dim
		problem_config.feature_dim = problem_config.feature_dim + 1 -- add one because the batcher adds one more feature 


		local train_batcher = SRLBatcher(train_labels_file,train_features_file,train_collision_file_base,batch_size, feature_dim, problem_config.max_predicates, problem_config.max_arguments, null_arc_index, params.use_cuda, true)
		
		problem_config.max_predicates = train_batcher.max_rows --todo: this isn't totally correct. It assumes that the biggest example in the training data is bigger than the biggest example in the test data
		problem_config.max_arguments = train_batcher.max_cols

		local test_batcher =  SRLBatcher(test_labels_file,test_features_file,test_collision_file_base,batch_size, feature_dim, problem_config.max_predicates, problem_config.max_arguments, null_arc_index, params.use_cuda, false)


		local y_shape = {batch_size,problem_config.max_predicates,problem_config.max_arguments,domain_size}

		problem_config.y_shape = y_shape
		local model = SRLSPEN(problem_config, params)

		local preprocess_func = nil
		local evaluator_factory = function(batcher, soft_predictor)
			local hard_predictor = RoundingPredictor(soft_predictor,y_shape)
			return SRLEvaluator(batcher, hard_predictor, problem_config.null_arc_index)
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

	local criterion_name = "ClassNLLCriterion"
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
	local gd_inference_config = GradientBasedInferenceConfig:get_gd_inference_config(params)
	
 	-- gd_inference_config.optimization_config.return_objective_values = true
	local gd_prediction_net = GradientBasedInference(y_shape, gd_inference_config):spen_inference(model)
	-- local par = nn.ParallelTable():add(nn.Identity()):add(nn.PrintNoNewline(true,false,'x',function(t) print(t:mean(1)) end))
	-- gd_prediction_net = nn.Sequential():add(gd_prediction_net):add(par):add(nn.SelectTable(1))
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
		local criterion_name = "ClassNLLCriterion"
		if(params.instance_weighted_loss == 1) then
			full_train_config.loss_wrapper = TrainingWrappers:instance_weighted_training(gd_prediction_net, criterion_name, y_shape, params)
		else
			full_train_config.loss_wrapper = TrainingWrappers:independent_training(gd_prediction_net, criterion_name, y_shape, params)
		end
	elseif(params.training_method == "SSVM") then
		local criterion_name = 'MSECriterion' --todo: surface an option for this
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
		opt_config = {learningRate=params.learning_rate}, --todo: unpack other command line args, like the adam parameters
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

	if(params.icnn == 1) then
		local params_to_clamp = model.global_potentials_net:parameters()
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

-- local function debug(pretrain_config, train_config) 
-- 	local classifier = pretrain_config.soft_predictor
-- 	local spen = train_config.soft_predictor
-- 	local iter = test_batcher:get_iterator()

-- 	-- local feats = model.features_network:forward(x)
-- 	-- local energy = model.energy_network:forward({y,feats})
-- 	-- print(energy)
-- 	local e_net = model.global_potentials_network
-- 	local ypp = nn.Sequential():add(nn.OneHot(36))

-- 	local y,x = unpack(iter())
-- 	local pred = spen:forward(x)
-- 	local yp = ypp:forward(y)
-- 	local f = model.features_network:forward(x)
-- 	local gt = e_net:forward({yp,f})
-- 	print(gt)
-- 	print(gt:mean())
-- 	local p = e_net:forward({pred,f})
-- 	print(p)
-- 	print(p:mean())
-- 	local peak = pred:max(4):mean()
-- 	print(peak)
-- 	-- local p1 = classifier:forward(x)
-- 	-- local p2 = spen:forward(x)
-- 	-- -- print(p1[1])
-- 	-- -- print(p2[1])
-- 	-- print(p1:max())
-- 	-- print(p2:max())
-- 	-- print((p1 - p2):abs():max())

-- end
-- debug(train_configurations.pretrain,train_configurations.full)
-- os.exit()

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



