require 'InferenceGrid'
local BPGrid, parent = torch.class('BPGrid', 'InferenceGrid')

function BPGrid:__init(y_shape, config)
	parent.__init(self,y_shape)
	self.config = config

	self.names = {'left','down','right','up'}
	self.set_complements = {}
	self.set_complements.left  = {'down','up'}
	self.set_complements.right = {'down','up'}
	self.set_complements.down  = {'left','right'}
	self.set_complements.up    = {'left','right'}

end

function BPGrid:inference_net()
	local log_node_potentials = nn.Identity()()
	local log_edge_potentials = nn.Identity()()

	--each of these is the same size as the node beliefs.
	local function init_zeros() return nn.Constant(torch.zeros(unpack(self.y_shape)))(log_node_potentials) end

	--Everything is in log space
	local messages = {}
	messages.down = init_zeros()
	messages.up = init_zeros()
	messages.right = init_zeros()
	messages.left = init_zeros()

	for t = 1,self.config.num_iters do
		for _, which_to_update in ipairs(self.names) do
			messages = self:update_messages(messages, which_to_update, log_node_potentials, log_edge_potentials)
		end
	end

	local beliefs = self:get_beliefs(messages,log_node_potentials)

	return nn.gModule({log_node_potentials,log_edge_potentials},{beliefs})
end

--these take nodes and return nodes
function BPGrid:get_beliefs(messages, log_node_potentials)
	local summands = {log_node_potentials}
	for _, n in ipairs(self.names) do
		table.insert(summands,messages[n])
	end

	local sum = nn.CAddTable()(summands)
	return self:softmax_4d()(sum)
end


--todo: document that log_edge_potentials is a grid
function BPGrid:update_messages(messages,which_to_update, log_node_potentials, log_edge_potentials)
	-- We update the messages coming into nodes from the <which_to_update> direction. 
	-- If we're updating the messages from nodes to the nodes above them, then the cavity looks to the left, down, and right.
	-- But, we want the *incoming* messages from those directions, so we use the current left, *up*, and right messages. 

	local cavity_names = self.set_complements[which_to_update]
	local cavity_messages = {log_node_potentials}

	--todo: need to be much more careful to make sure these are evaluated at the right places. there is probably an off by one error

	for _, direction in ipairs(cavity_names) do
		local message_from_neighbor = messages[direction]
		table.insert(cavity_messages, message_from_neighbor)
	end

	local non_recurrent_cavity_messages = nn.CAddTable()(cavity_messages)

	--now we do a sweep, updating messages[which_to_update] iteratively
	local relevant_edge_potentials = self:get_edge_potentials(log_edge_potentials,which_to_update)

	local vertical = which_to_update == "up" or which_to_update == "down"
	local size = vertical and self.height  or self.width
	local axis = vertical and 1 or 2
	local index_backwards = (name == 'right' or name == 'up')

	local slice_shape, pad_shape, expansion_shape
	if(vertical) then
		slice_shape = {self.batch_size,1,self.width,self.domain_size,false}
		pad_shape = {self.batch_size,self.width,self.domain_size}
		expansion_shape = {self.batch_size,self.width,1,self.domain_size,false}
	else
		slice_shape = {self.batch_size,self.height,1,self.domain_size,false}
		pad_shape = {self.batch_size,self.height,self.domain_size}
		expansion_shape = {self.batch_size,self.height,1,self.domain_size,false}

	end


	local new_message_table = {}

	local function insert_msg(msg)
		local msg_expand = nn.Reshape(unpack(slice_shape))(msg)
		if(index_backwards) then
			--if index_backwards, then you want to build up the list as first-in-last out
			table.insert(new_message_table,1,msg_expand)
		else
			table.insert(new_message_table,msg_expand)
		end
	end

	local padding =  nn.Constant(torch.zeros(unpack(pad_shape)))(log_node_potentials)
	insert_msg(padding)

	local prev_msg = padding
	--At i = i, it updates the messages coming out of the ith thing into the (i+1)th thing.

	for i = 1,(size-1) do
		local sliced_cavity = nn.Select(axis+1,i)(non_recurrent_cavity_messages)
		local sliced_edge_potentials = nn.Select(axis+1,i)(relevant_edge_potentials)

		local full_cavity = nn.CAddTable()({sliced_cavity, prev_msg})
		local cavity_expand = nn.Reshape(unpack(expansion_shape))(full_cavity)
		local cavity_replicate = nn.Replicate(self.domain_size, 4)(cavity_expand)
		
		--the next two steps do a matrix multiply in log space
		local sum = nn.CAddTable()({cavity_replicate,sliced_edge_potentials})
		local msg = nn.LogSumExp(4)(sum)

		-- --normalize messages, in log space
		local logZs = nn.LogSumExp(4)(msg)
		msg = nn.CSubTable()({msg,logZs})

		insert_msg(msg)
		prev_msg = msg
	end

	local new_message = nn.JoinTable(axis+1)(new_message_table)

	local updated_messages = {}
	for _, n in ipairs(self.names) do
		if(n == which_to_update) then
			updated_messages[n] = new_message
		else
			updated_messages[n] = messages[n]
		end
	end

	return updated_messages
end

function BPGrid:get_edge_potentials(edge_potentials,name)
	local vertical = name == 'up' or name == 'down'
	local axis = vertical and 1 or 2
	local index_backwards = (name == 'right' or name == 'down')

	local to_return = nn.SelectTable(axis)(edge_potentials)

	-- Above, we always do a left multiply with the cavity message, 
	-- so we need to transpose if we should've been doing a right multiply.
	if(index_backwards) then
		to_return = nn.Transpose({4,5})(to_return)
	end

	return to_return
end
