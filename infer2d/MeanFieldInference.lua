require 'InferenceGrid'
local MeanFieldInferenceGrid, parent = torch.class('MeanFieldInferenceGrid', 'InferenceGrid')

function MeanFieldInferenceGrid:__init(y_shape, config)
	parent.__init(self,y_shape)
	self.config = config
end

function MeanFieldInferenceGrid:inference_net()
	local log_node_potentials = nn.Identity()()
	local log_edge_potentials = nn.Identity()()

	local init_node_logits = nn.Constant(torch.zeros(unpack(self.y_shape)))(log_edge_potentials)
	local init_node_beliefs = self:softmax_4d()(init_node_logits)

	local node_beliefs = init_node_beliefs
	for i = 1,self.config.num_iters do
		node_beliefs = self:mean_field_update(node_beliefs,log_node_potentials,log_edge_potentials)
	end

	return nn.gModule({log_node_potentials,log_edge_potentials},{node_beliefs})
end

function MeanFieldInferenceGrid:mean_field_update(node_beliefs, log_node_potentials, log_edge_potentials)
	-- node_beliefs: b x l x d

	local sizes = {self.height,self.width}

	--axis: (1) vertical or (2) horizontal
	--use_lower_index: if true, then compute contribution to node beliefs from edge that has a *higher* index than the node (ie below for axis == 1 and to the right for axis == 2)
	local function contribution_from_neighbor(axis,use_lower_index)
		--first, slice off the relevant nodes. 
		
		local h = axis == 1 and self.height - 1 or self.height
		local w = axis == 2 and self.width - 1  or self.width
		local num_elements = self.batch_size*h*w

		local relevant_edge_potentials = nn.SelectTable(axis)(log_edge_potentials)
		local reshaped_edge_potentials = nn.Reshape(self.batch_size*h*w,self.domain_size,self.domain_size, false)(relevant_edge_potentials)

		local start = use_lower_index and 1 or 2
		local sliced_node_beliefs = nn.Narrow(axis+1,start,sizes[axis]-1)(node_beliefs) --todo: we prob want an optional transpose here as well
		local reshaped_node_beliefs = nn.Reshape(num_elements, 1, self.domain_size, false)(sliced_node_beliefs)

		local transA = false
		local transB = not use_lower_index
		local prod = nn.MM(transA,transB)({reshaped_node_beliefs,reshaped_edge_potentials})
		prod = nn.Reshape(self.batch_size,h,w,self.domain_size,false)(prod)

		--then, pad the result with zeros on either the beginning or the end
		local padding
		if(axis == 1) then
			padding = nn.Constant(torch.zeros(self.batch_size,1,self.width,self.domain_size))(node_beliefs) 
		else
			padding = nn.Constant(torch.zeros(self.batch_size,self.height,1,self.domain_size))(node_beliefs) 
		end
		if(use_lower_index) then
			return nn.JoinTable(axis+1)({prod,padding})
		else
			return nn.JoinTable(axis+1)({padding,prod})
		end

	end

	local summands = {log_node_potentials}
	for axis = 1,2 do
		for _, use_lower_index in ipairs({true,false}) do
			table.insert(summands,contribution_from_neighbor(axis,use_lower_index))
		end
	end


	local combined_contributions = nn.CAddTable()(summands)
	return self:softmax_4d()(combined_contributions)
end
