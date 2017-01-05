require 'SPEN'
local ChainCRF, parent = torch.class('ChainCRF','SPEN')
package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'
require 'SumProductInference'
require 'ViterbiInference'


function ChainCRF:__init(config, params)
	self.config = config
	assert(config.y_shape)
	self.y_shape = config.y_shape
	self.batch_size, self.length, self.domain_size = unpack(config.y_shape)
	assert(self.length > 2)
	self.feature_size = config.feature_size

	parent.__init(self, config, params)
	self.log_edge_potentials_network = self:log_edge_potentials_net()
	self.energy_from_potentials_network = self:score_net()
end

function ChainCRF:normalize_unary_prediction()
	return Inference1DUtil:softmax_3d(self.config.y_shape)
end

function ChainCRF:features_net(x)
	local x = nn.Identity()()
	local x_pad
	if(self.config.feature_width == 1) then
		x_pad = x
	else
		local pad_size = 0.5*(self.config.feature_width - 1)
		local left_pad = nn.Constant(torch.zeros(self.batch_size,pad_size,self.feature_size))(x)
		local right_pad = nn.Constant(torch.zeros(self.batch_size,pad_size,self.feature_size))(x)
		x_pad = nn.JoinTable(2,3)({left_pad,x,right_pad})
	end

	local f1 = nn.TemporalConvolution(self.feature_size,self.config.feature_hid_size,self.config.feature_width,1)(x_pad)
	local features = nn.ReLU()(f1)
	return nn.gModule({x},{features})
end

function ChainCRF:unary_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local feature_size = self.config.feature_hid_size
	local local_potentials = nn.TemporalConvolution(feature_size,self.domain_size,1,1)(conditioning_values)
	local_potentials.data.module.weight:mul(self.config.local_potentials_scale)
	local local_potentials_net = nn.gModule({conditioning_values},{local_potentials})
	self.local_potentials_net = local_potentials_net
	return local_potentials_net
end

function ChainCRF:pairwise_potentials()
	local conditioning_values = nn.Identity()() --b x l x h
	local feature_size = self.config.feature_hid_size

	local f_for_pairwise = conditioning_values
	if(self.config.data_independent_transitions) then
		f_for_pairwise = nn.MulConstant(0)(f_for_pairwise)
	end

	local pairwise_potentials = nn.TemporalConvolution(feature_size,self.domain_size*self.domain_size,2,1)(f_for_pairwise)
	pairwise_potentials.data.module.weight:mul(self.config.pairwise_potentials_scale)
	pairwise_potentials = nn.Reshape(self.batch_size,self.length-1,self.domain_size,self.domain_size,false)(pairwise_potentials)

	local pairwise_potentials_net = nn.gModule({conditioning_values},{pairwise_potentials})
	self.pairwise_potentials_net = pairwise_potentials_net 

	return pairwise_potentials_net
end

function ChainCRF:global_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local y = nn.Identity()()

	local clique_margs = Inference1DUtil:make_outer_product_marginals(y,self.y_shape)
	local pairwise_potentials = self:pairwise_potentials()(conditioning_values)

	local mul = nn.CMulTable()({clique_margs,pairwise_potentials})
	local energy = nn.Sum(2)(nn.Reshape(-1)(mul))

	return nn.gModule({y,conditioning_values},{energy})
end

function ChainCRF:score_net()
	local pairwise_potentials = nn.Identity()() --b x l x h
	local y = nn.Identity()()

	local clique_margs = Inference1DUtil:make_outer_product_marginals(y,self.y_shape)

	local mul = nn.CMulTable()({clique_margs,pairwise_potentials})
	local energy = nn.Sum(2)(nn.Reshape(-1)(mul))
	energy = nn.MulConstant(-1,true)(energy)
	return nn.gModule({y,pairwise_potentials},{energy})
end

--this takes x and returns the potentials of the MRF 
function ChainCRF:log_edge_potentials_net()

	local x = nn.Identity()()
	local features = self.features_network(x)


	local merged_edge_potentials = self:merge_local_potentials_with_edge_potentials(self.local_potentials_net(features), self.pairwise_potentials_net(features))
	merged_edge_potentials = nn.MulConstant(-1)(merged_edge_potentials) --we multiply by -1 because the SPEN inference performs energy minimization, but our CRF inference algorithms expect energy maximization.
	local log_edge_potentials_net = nn.gModule({x},{merged_edge_potentials})
	return log_edge_potentials_net
end



function ChainCRF:merge_local_potentials_with_edge_potentials(local_potentials, pairwise_potentials)

	local to_add_to_left = nn.Narrow(2,1,self.length-1)(local_potentials)
	to_add_to_left = nn.Replicate(self.domain_size,4,4)(to_add_to_left)

	local pairwise_potentials_with_left = nn.CAddTable()({pairwise_potentials,to_add_to_left})

	--now split off the first l-2 blocks, which we won't modify
	local left_blocks = nn.Narrow(2,1,self.length-2)(pairwise_potentials_with_left)

	local right_block = nn.Narrow(2,self.length-1,1)(pairwise_potentials_with_left)
	local to_add_to_right = nn.Narrow(2,self.length,1)(local_potentials)
	to_add_to_right = nn.Replicate(self.domain_size,3,4)(to_add_to_right)
	local modified_right_block = nn.CAddTable()({right_block,to_add_to_right})

	local merged_potentials = nn.JoinTable(2,4)({left_blocks,modified_right_block})

	return merged_potentials

end


function ChainCRF:predict(x)
	self.map_inference = self.map_inference or ViterbiInference(self.y_shape)

	local log_edge_potentials_value = self.log_edge_potentials_network:forward(x)
	return self.map_inference:predict(log_edge_potentials_value)
end

function ChainCRF:predict_mbr(x)
	self.inference = self.inference or SumProductInference(self.y_shape)

	local log_edge_potentials_value = self.log_edge_potentials_network:forward(x)
	local edge_marginals, logZ = self.inference:infer_values(log_edge_potentials_value)
	local node_marginals = Inference1DUtil:edge_marginals_to_node_marginals(edge_marginals,self.y_shape)
	print(node_marginals[1])
	local values, inds = node_marginals:max(3)
	return inds:reshape(self.batch_size,self.length)
end

function ChainCRF:sample_from_joint(x)
	self.inference = self.inference or SumProductInference(self.y_shape)

	local log_edge_potentials_value = self.log_edge_potentials_network:forward(x)
	local edge_marginals, logZ = self.inference:infer_values(log_edge_potentials_value)
	local node_marginals = Inference1DUtil:edge_marginals_to_node_marginals(edge_marginals,self.y_shape)
	local sampled_inds = torch.Tensor(self.batch_size,self.length)
	local function sample_from_multinomial(probs)
		assert(probs:nDimension() == 2)
		assert(probs:sum(2):add(-1):lt(0.0001):all())
		return torch.multinomial(probs,1,false):reshape(self.batch_size)
	end

	local function normalize_by_column_in_place(t)
		local batch_size = t:size()[1]
		local domain_size = t:size()[2]
		local sums = torch.sum(t,2):reshape(batch_size,1):expandAs(t)
		t:cdiv(sums)
		return t
	end
	--sample the left value from its marginals
	local sampled_left_inds = sample_from_multinomial(node_marginals:narrow(2,1,1):reshape(self.batch_size,self.domain_size))
	sampled_inds:narrow(2,1,1):copy(sampled_left_inds)
	local inds_at_prev_timestep = sampled_left_inds
	self.batch_mat_mul = self.batch_mat_mul or nn.MM()
	--then sample the next value from the conditional marginal
	for idx = 1,(self.length-1) do
		local relevant_marginals = edge_marginals:narrow(2,idx,1):reshape(self.batch_size,self.domain_size,self.domain_size)

		--torch gather is very confusing. just use matrix multiply instead to slice the edge marginals when conditioning
		local one_hot_inds = torch.Tensor(self.batch_size,1,self.domain_size):zero()
		for i = 1,self.batch_size do
			local idx2 = inds_at_prev_timestep[i]
			one_hot_inds[i][1][idx2] = 1
		end
		sliced_marginals = self.batch_mat_mul:forward({one_hot_inds,relevant_marginals}):reshape(self.batch_size,self.domain_size)
		local cond_probs = normalize_by_column_in_place(sliced_marginals)
		local inds_at_timestep = sample_from_multinomial(cond_probs)
		sampled_inds:narrow(2,idx+1,1):copy(inds_at_timestep)
		inds_at_prev_timestep = inds_at_timestep
	end
	return sampled_inds:reshape(self.batch_size,self.length,1)
end