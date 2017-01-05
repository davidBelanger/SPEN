package.path = package.path .. ';model/?.lua'
require 'ChainCRF'
require 'SequenceTagging'

local ChainCRFSequenceTagging, parent = torch.class('ChainCRFSequenceTagging','SequenceTagging')


--this doesn't load data from a file. Instead it generates training examples on the fly by drawing from a fixed CRF model

function ChainCRFSequenceTagging:__init(model_config, dataset_config)
	self.model_config = model_config
	self.dataset_config = dataset_config
	self.crf_model = ChainCRF(model_config)
	self.inference = SumProductInference(model_config.y_shape)

	parent.__init(self,dataset_config)
end

function ChainCRFSequenceTagging:get_train_batcher()
	return BatcherFromFactory(function() return self:sample_batch() end, self.dataset_config.preprocess_train,self.dataset_config.num_train_batches,true)
end

function ChainCRFSequenceTagging:get_test_batcher()
	return BatcherFromFactory(function() return self:sample_batch() end, self.dataset_config.preprocess_test, self.dataset_config.num_test_batches)
end

function ChainCRFSequenceTagging:sample_batch()
	local x = self:sample_x()
	local y = self.crf_model:sample_from_joint(x)
	return {y,x}
end


function ChainCRFSequenceTagging:sample_x()
	return torch.randn(self.model_config.batch_size,self.model_config.length,self.model_config.feature_size)
end
