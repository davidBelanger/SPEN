package.path = package.path .. ';model/?.lua'
require 'LinkPrediction'

local SimpleBinaryLinkPrediction, parent = torch.class('SimpleBinaryLinkPrediction','LinkPrediction')


function SimpleBinaryLinkPrediction:__init(model_config, dataset_config)
	self.dataset_config = dataset_config
	self.model_config = model_config
	assert(model_config.domain_size == 2)
	assert(self.dataset_config)
	self:init_graph_model(model_config)

	parent.__init(self,dataset_config)

end

function SimpleBinaryLinkPrediction:init_graph_model(model_config)
	local U = torch.randn(1,model_config.architecture.hid_feature_size,model_config.feature_size)
	local V = torch.randn(1,model_config.architecture.hid_feature_size,model_config.feature_size)
	self.model = {
		U = U,
		V = V
	}
end

function SimpleBinaryLinkPrediction:get_logits(x)
	local xt = x:transpose(2,3)
	local function expand_to_batch(t) return t:expand(self.model_config.batch_size,t:size(2),t:size(3)) end
	local U_tile = expand_to_batch(self.model.U)
	local V_tile = expand_to_batch(self.model.V)
	local child_embeddings = torch.bmm(U_tile, xt) 
	local parent_embeddings = torch.bmm(V_tile, xt) 

	local scores = torch.bmm(child_embeddings:transpose(2,3),parent_embeddings)

	return scores
end

function SimpleBinaryLinkPrediction:predict(x)
	local scores = self:get_logits(x)
	local probs = torch.sigmoid(scores)
	local prediction = probs:gt(0.5):long():add(1)
	return prediction
end

function SimpleBinaryLinkPrediction:sample_from_model(x)
	local scores = self:get_logits(x)
	local probs = torch.sigmoid(scores)
	local unif = torch.rand(probs:size())
	--we add 1 so that 1 = not observed and 2 = observed. 
	local adjacency_matrix = probs:gt(unif):long():add(1)
	return adjacency_matrix
end

function SimpleBinaryLinkPrediction:init_test_batches()
	for i = 1,self.dataset_config.num_test_batches do
		table.insert(self.test_batches,self:get_batch())
	end
	return self.dataset_config.num_test_batches
end

--this corresponds to an infinitely big training set
function SimpleBinaryLinkPrediction:get_train_batch()
	local b = self:get_batch()

	if(self.dataset_config.preprocess_train) then 
		return self.dataset_config.preprocess_train(b)
	else
		return b
	end
end

function SimpleBinaryLinkPrediction:get_batch()
	local x = self:sample_x()
	local y = self:sample_from_model(x)
	return {y,x}
end


function SimpleBinaryLinkPrediction:sample_x()
	return torch.randn(self.model_config.batch_size,self.model_config.num_nodes,self.model_config.feature_size)
end


