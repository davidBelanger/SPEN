require 'Inference1DUtil'
local MeanFieldInference, parent = torch.class('MeanFieldInference')

function MeanFieldInference:__init(y_shape, config)
	self.config = config
	self.y_shape = y_shape
	self.batch_size,self.length, self.domain_size = unpack(self.y_shape)

end

function MeanFieldInference:inference_net()
	local log_edge_potentials = nn.Identity()()

	local init_node_logits = nn.Constant(torch.zeros(unpack(self.y_shape)))(log_edge_potentials)
	local init_node_beliefs = Inference1DUtil:softmax_3d(self.y_shape)(init_node_logits)
	init_node_beliefs = nn.View(self.batch_size,self.length, self.domain_size, 1)(init_node_beliefs)

	local node_beliefs = init_node_beliefs
	for i = 1,self.config.num_iters do
		node_beliefs = self:mean_field_update(node_beliefs,log_edge_potentials)
	end

	return nn.gModule({log_edge_potentials},{node_beliefs})
end

function MeanFieldInference:mean_field_update(node_beliefs, log_edge_potentials)
	-- node_beliefs: b x l x d
	local node_beliefs_except_left = nn.Narrow(2,2,self.length-1)(node_beliefs)
	local node_beliefs_except_right = nn.Narrow(2,1,self.length-1)(node_beliefs)
	

	local function parallel_MM(x,M,transB)
		local x_reshape = nn.Reshape(self.batch_size*(self.length-1), 1, self.domain_size, false)(x)
		local M_reshape = nn.Reshape(self.batch_size*(self.length-1), self.domain_size, self.domain_size, false)(M)

		local prod = nn.MM(transA,transB)({x_reshape,M_reshape})
		return nn.Reshape(self.batch_size, self.length-1, self.domain_size, 1, false)(prod)
	end

	local contributions_from_left  = parallel_MM(node_beliefs_except_right,  log_edge_potentials, false)
	local contributions_from_right = parallel_MM(node_beliefs_except_left, log_edge_potentials, true)

	local function padding() return nn.Constant(torch.zeros(self.batch_size,1,self.domain_size))(log_edge_potentials) end
	contributions_from_left = nn.JoinTable(2)({padding(), contributions_from_left})
	contributions_from_right = nn.JoinTable(2)({contributions_from_right, padding()})
	
	local combined_contributions = nn.CAddTable()({contributions_from_left, contributions_from_right})
	return Inference1DUtil:softmax_3d(self.y_shape)(combined_contributions)
end
