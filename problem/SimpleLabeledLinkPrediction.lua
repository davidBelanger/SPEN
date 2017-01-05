package.path = package.path .. ';model/?.lua'
require 'LinkPrediction'
require 'Inference1DUtil'
local SimpleLabeledLinkPrediction, parent = torch.class('SimpleLabeledLinkPrediction','LinkPrediction')


function SimpleLabeledLinkPrediction:__init(model_config, dataset_config)
	self.dataset_config = dataset_config
	self.model_config = model_config
	assert(self.dataset_config)
	self:init_graph_model(model_config)

	parent.__init(self,dataset_config)

end

function SimpleLabeledLinkPrediction:init_graph_model(model_config)
	local U = torch.randn(1,model_config.architecture.hid_feature_size*model_config.domain_size,model_config.feature_size)
	local V = torch.randn(1,model_config.architecture.hid_feature_size*model_config.domain_size,model_config.feature_size)
	self.model = {
		U = U,
		V = V
	}
end

function SimpleLabeledLinkPrediction:get_logits(x)
	local xt = x:transpose(2,3)
	local function expand_to_batch(t) return t:expand(self.model_config.batch_size,t:size(2),t:size(3)) end
	local U_tile = expand_to_batch(self.model.U)
	local V_tile = expand_to_batch(self.model.V)
	local num_nodes = self.model_config.num_nodes
	local domain_size = self.model_config.domain_size
	local architecture = self.model_config.architecture
	local batch_size = self.model_config.batch_size
	local child_embeddings = torch.bmm(U_tile, xt) -- b x (h x d) x n
	child_embeddings = child_embeddings:view(batch_size*domain_size,architecture.hid_feature_size,num_nodes) -- (b x d) x h x n
	local parent_embeddings = torch.bmm(V_tile, xt) 
	parent_embeddings = parent_embeddings:view(batch_size*domain_size,architecture.hid_feature_size,num_nodes) -- (b x d) x h x n


	local scores = torch.bmm(child_embeddings:transpose(2,3),parent_embeddings)
	scores = scores:view(batch_size,domain_size,num_nodes,num_nodes)
	scores = scores:transpose(2,3):transpose(3,4)
	return scores
end

function SimpleLabeledLinkPrediction:sample_from_model(x)
	local num_nodes = self.model_config.num_nodes
	local domain_size = self.model_config.domain_size
	local architecture = self.model_config.architecture
	local batch_size = self.model_config.batch_size

	local scores = self:get_logits(x)
	self.softmax_4d = self.softmax_4d or Inference1DUtil:softmax_4d(self.model_config.y_shape)

	scores = scores:contiguous()
	local probs = self.softmax_4d:forward(scores)
	local probs_reshape = probs:view(batch_size*num_nodes*num_nodes,domain_size)
	local inds = torch.multinomial(probs_reshape,1)
	labeled_adjacency_matrix = inds:view(batch_size,num_nodes,num_nodes)
	return labeled_adjacency_matrix
end

function SimpleLabeledLinkPrediction:predict(x)
	local scores = self:get_logits(x)
	local values, inds = scores:max(scores:nDimension())
	local pred = inds:select(inds:nDimension(),1)
	return pred
end

function SimpleLabeledLinkPrediction:init_test_batches()
	for i = 1,self.dataset_config.num_test_batches do
		table.insert(self.test_batches,self:get_batch())
	end
	return self.dataset_config.num_test_batches
end

--this corresponds to an infinitely big training set
function SimpleLabeledLinkPrediction:get_train_batch()
	local b = self:get_batch()

	if(self.dataset_config.preprocess_train) then 
		return self.dataset_config.preprocess_train(b)
	else
		return b
	end
end

function SimpleLabeledLinkPrediction:get_batch()
	local x = self:sample_x()
	local y = self:sample_from_model(x)
	return {y,x}
end


function SimpleLabeledLinkPrediction:sample_x()
	return torch.randn(self.model_config.batch_size,self.model_config.num_nodes,self.model_config.feature_size)
end




