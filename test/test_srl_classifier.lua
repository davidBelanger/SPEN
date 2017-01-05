

local seed = 12345
torch.manualSeed(seed)

require 'Imports'

local params = {
	batch_size = 128,
	use_cuda = false,
	gradient_clip = 2.0,
	learning_rate = 0.001,
	num_epochs = 2000,
	batches_per_epoch = 500,
	evaluation_frequency = 10,
	null_arc_index = 1,
	learning_rate_decay = 0.001
}
local train_file = "srl/processed_data/conll2005/train.classification.torch"
local test_file = "srl/processed_data/conll2005/dev.classification.torch"
local feature_size = 128
local label_domain_size = 36
local y_shape = {params.batch_size,label_domain_size}

local classifier = nn.Sequential():add(nn.Linear(feature_size,label_domain_size)):add(nn.SoftMax())
local preprocess_func = nil
local train_batcher = BatcherFromFile({train_file}, preprocess_func, params.batch_size, params.use_cuda)
local test_batcher  = BatcherFromFile({test_file},  preprocess_func, params.batch_size, params.use_cuda)

local hard_predictor = RoundingPredictor(classifier,y_shape)
local evaluator = SRLEvaluator(test_batcher, hard_predictor, params.null_arc_index)


local criterion_name = "ClassNLLCriterion"
local loss_wrapper = Config:independent_training(classifier, criterion_name, y_shape, params)


local optimization_config = {
	opt_state = {},
	opt_config = {learningRate=params.learning_rate, learningRateDecay=params.learning_rate_decay}, --todo: unpack other command line args, like the adam parameters
	opt_method = optim.adam,
	gradient_clip = params.gradient_clip,
	regularization = nil,
	modules_to_update = classifier
}

local general_config = {
	num_epochs = params.num_epochs,
	batches_per_epoch = params.batches_per_epoch,
	batch_size = params.batch_size,
	assert_nan = true,
}

local callbacks = {}
local evaluate = Callback(function(data) return evaluator:evaluate(data.epoch) end, params.evaluation_frequency)
table.insert(callbacks,evaluate)

Train(loss_wrapper,train_batcher, optimization_config, general_config, callbacks):train()
