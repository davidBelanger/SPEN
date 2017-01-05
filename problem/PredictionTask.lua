local PredictionTask = torch.class('PredictionTask')

function PredictionTask:__init(config)

end

function PredictionTask:get_test_batcher(preprocess)
	assert(false,'abstract method')
end

function PredictionTask:get_train_batcher(preprocess)
	assert(false,'abstract method')
end



