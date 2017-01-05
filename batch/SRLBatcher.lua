local SRLBatcher = torch.class('SRLBatcher')

function SRLBatcher:__init(labels_file,features_file,collision_file_base,batch_size,feature_dim,max_rows,max_cols,null_arc_index, cuda, shrink_to_fit)
	self.batch_size = batch_size
	self.feature_dim = feature_dim 
	self.expanded_feature_dim = self.feature_dim + 1
	self.null_arc_index = null_arc_index
	self.cuda = cuda

	assert(feature_dim == 128)
	self.arc_feature_dim = 128
	self.node_feature_dim = 64

	print('loading from '..labels_file)
	self.labels = torch.load(labels_file)
	self.num_examples = #self.labels
	print('loading from '..features_file)

	self.all_features = torch.load(features_file)
	self.argument_features = self.all_features:narrow(2,1,64)
	self.predicate_features = self.all_features:narrow(2,65,64)
	self.arc_features = self.all_features:narrow(2,129,128)
	self.one_pass_taken = false

	self.sparse_collisions = {}
	for _, ext in ipairs({"p2p","p2a","a2a"}) do
		local f = collision_file_base.."."..ext
		print('loading from: '..f)
		table.insert(self.sparse_collisions,torch.load(f))
	end


	self.cur_idx = 1
	local function find_max(column)
		local max = -1
		for i = 1,#self.labels do
			max = math.max(max,self.labels[i]:select(2,column):max())
		end
		return max
	end
	if(shrink_to_fit) then
		self.max_rows = math.min(max_rows,find_max(2))
		self.max_cols = math.min(max_cols,find_max(3))
		print('using dense labels of size: '..self.max_rows.." x "..self.max_cols)
	else
		self.max_cols = max_cols
		self.max_rows = max_rows
	end
	self:initialize_batch_elements()
end

function SRLBatcher:usable_example(index)
	local labs = self.labels[index]
	return labs:select(2,2):max() <= self.max_rows and labs:select(2,3):max() <= self.max_cols

end

function SRLBatcher:initialize_batch_elements()
	self.preallocated_features = {}
	self.preallocated_labels = {}
	self.preallocated_collisions = {}
	self.preallocated_indicators = {}


	for i = 1,self.batch_size do
		local dense_labels = torch.Tensor(self.max_rows,self.max_cols):fill(self.null_arc_index) 
		table.insert(self.preallocated_labels,dense_labels)

		local indicators = torch.Tensor(self.max_rows,self.max_cols):zero()
		table.insert(self.preallocated_indicators,indicators)

		local dense_features1 = torch.Tensor(self.max_rows, self.max_cols, self.expanded_feature_dim):zero() 
		local dense_features2 = torch.Tensor(self.max_rows, self.node_feature_dim):zero() 
		local dense_features3 = torch.Tensor(self.max_cols, self.node_feature_dim):zero() 
		local features = {dense_features1, dense_features2, dense_features3}
		table.insert(self.preallocated_features,features)

		local collisions1 = torch.Tensor(self.max_rows, self.max_rows):zero() 
		local collisions2 = torch.Tensor(self.max_rows, self.max_cols):zero() 
		local collisions3 = torch.Tensor(self.max_cols, self.max_cols):zero() 
		local collisions = {collisions1,collisions2,collisions3}

		table.insert(self.preallocated_collisions,collisions)

	end
end


function SRLBatcher:expand_dim_1(t)
	local s = t:size()
	local ss = {1}
	for i = 1,#s do
		table.insert(ss,s[i])
	end
	return t:view(unpack(ss))
end

function SRLBatcher:get_next_batch()
	local batch_labels = {}
	local batch_features = {{},{},{}}
	local batch_collisions = {{},{},{}}
	local batch_indicators = {}
	local num_actual_data = 0
	local end_reached = false

	for i = 1,self.batch_size do
		local dense_features = self.preallocated_features[i]
		local dense_labels = self.preallocated_labels[i]
		local dense_collisions = self.preallocated_collisions[i]

		local filtered_arc_indicators = self.preallocated_indicators[i]
		local reached_end_in_this_batch = self:get_next_example(dense_features, dense_labels, dense_collisions, filtered_arc_indicators)
		if(not end_reached) then num_actual_data = num_actual_data + 1 end
		end_reached = end_reached or reached_end_in_this_batch

		local feats = Util:deep_apply(dense_features,function(t) return self:expand_dim_1(t) end)
		local collisions = Util:deep_apply(dense_collisions,function(t) return self:expand_dim_1(t) end)

		local indicators = filtered_arc_indicators:view(1,self.max_rows,self.max_cols)
		table.insert(batch_indicators,indicators)
		table.insert(batch_labels,dense_labels:view(1,self.max_rows,self.max_cols))
		for i = 1,3 do
			table.insert(batch_features[i],feats[i])
			table.insert(batch_collisions[i],collisions[i])
		end
	end
	
	self.join_labels   = self.join_labels or nn.JoinTable(1)
	self.join_indicators = self.join_indicators or nn.JoinTable(1)
	self.join_features = self.join_features or nn.ParallelTable():add(nn.JoinTable(1)):add(nn.JoinTable(1)):add(nn.JoinTable(1))
	self.join_collisions = self.join_collisions or nn.ParallelTable():add(nn.JoinTable(1)):add(nn.JoinTable(1)):add(nn.JoinTable(1))

	local feats_and_collisions = {self.join_features:forward(batch_features),self.join_collisions:forward(batch_collisions)}
	local merged_features = {feats_and_collisions, self.join_indicators:forward(batch_indicators)}
	local labels = self.join_labels:forward(batch_labels)

	--todo: we could be copying into pre-allocated cuda locations
	if(self.cuda) then
		merged_features = Util:deep_apply(merged_features,function(t) return t:cuda() end) 
		labels = labels:cuda()
	end
	if(num_actual_data == 0) then return {nil,nil,0} end
	assert(num_actual_data > 0, 'should not be returning a batch with no actual data')
	return {labels, merged_features, num_actual_data, end_reached}
end

-- --TODO: this could definitely be improved
-- function SRLBatcher:slow_scatter(row_indices,col_indices,features,dense_features, labels, dense_labels, filtered_arc_indicators)
-- 	for i = 1,col_indices:size(1) do
-- 		local row_index = row_indices[i]
-- 		local col_index = col_indices[i]
-- 		dense_features[row_index][col_index]:narrow(1,1,self.feature_dim):copy(features[i]) --copy over the first self.feature_dim features
-- 		dense_features[row_index][col_index][self.expanded_feature_dim] = 0 --set the final feature to 0, since it is a candidate edge
-- 		dense_labels[row_index][col_index] = labels[i]
-- 		filtered_arc_indicators[row_index][col_index] = 1.0
-- 	end
-- end



function SRLBatcher:init_tracking_of_rejcted_examples()
	self.seen_rejected_examples = {}
	self.num_rejected_examples = 0
	self.total_rejected_positive_arcs = 0.0
	self.track_rejected_examples = true
end
function SRLBatcher:record_rejected_example(index)
	if(self.seen_rejected_examples[index]) then
		print(self.seen_rejected_examples)
		print(#self.seen_rejected_examples)
		print(index)
	end
	assert(not self.seen_rejected_examples[index],'should not be recording a rejected example twice: '..index)
	self.seen_rejected_examples[index] = true
	self.num_rejected_examples = self.num_rejected_examples + 1
	local arc_labels = self.labels[index]:select(2,4)
	self.total_rejected_positive_arcs = self.total_rejected_positive_arcs + arc_labels:ne(self.null_arc_index):sum()*1.0
end

function SRLBatcher:report_on_rejected_examples()
	local report = {
		total_rejected_positive_arcs = self.total_rejected_positive_arcs,
		total_rejected_examples = self.num_rejected_examples
	}
	return report
end

function SRLBatcher:get_next_example(dense_features,dense_labels, dense_collisions, filtered_arc_indicators)
	local end_reached = self.cur_idx == self.num_examples

	if(self.cur_idx > self.num_examples) then 
		self.cur_idx = 1 
		self.one_pass_taken = true
	end
	while(not self:usable_example(self.cur_idx)) do
		if(self.track_rejected_examples and (not self.one_pass_taken)) then self:record_rejected_example(self.cur_idx) end
		self.cur_idx = self.cur_idx + 1

		if(self.cur_idx > self.num_examples) then 
			self.cur_idx = 1 
			end_reached = true
			self.one_pass_taken = true
		end
	end

	local labs = self.labels[self.cur_idx]

	local len = labs:size(1)
	local line_start = labs[1][5]
	local line_end = labs[len][5]

	for i = 1,3 do
		dense_features[i]:zero()
		dense_collisions[i]:zero()
	end

	assert(line_end == line_start + len - 1)
	local features = self.arc_features:narrow(1,line_start,len)
	local row_indices = labs:select(2,2)
	local col_indices = labs:select(2,3)
	local labels = labs:select(2,4)

	dense_labels:fill(self.null_arc_index) --by default, the labels not in the candidates file are considered null arcs
	dense_features[1]:select(3,self.expanded_feature_dim):fill(1.0) --we set a special feature to 1 for all edges that weren't in the candidates file
	filtered_arc_indicators:fill(0.0)


	self:slow_scatter(row_indices,col_indices,features, dense_features[1], labels, dense_labels, filtered_arc_indicators)
	self:slow_scatter2(row_indices, dense_features[2], self.predicate_features:narrow(1,line_start,len))
	self:slow_scatter2(col_indices, dense_features[3], self.argument_features:narrow(1,line_start,len))

	for i = 1,3 do
		local sparse_collisions = self.sparse_collisions[i][self.cur_idx]

		if(sparse_collisions:dim() > 0) then
			self:slow_scatter3(sparse_collisions,dense_collisions[i])
		end
		assert(dense_collisions[i]:dim() == 2)
	end
	self.cur_idx = self.cur_idx + 1

	return end_reached
end


function SRLBatcher:slow_scatter(row_indices,col_indices,features,dense_features, labels, dense_labels, filtered_arc_indicators)
	for i = 1,col_indices:size(1) do
		local row_index = row_indices[i]
		local col_index = col_indices[i]
		dense_features[row_index][col_index]:narrow(1,1,self.feature_dim):copy(features[i]) --copy over the first self.feature_dim features
		dense_features[row_index][col_index][self.expanded_feature_dim] = 0 --set the final feature to 0, since it is a candidate edge
		dense_labels[row_index][col_index] = labels[i]
		filtered_arc_indicators[row_index][col_index] = 1.0
	end
end

--todo: check this
function SRLBatcher:slow_scatter2(inds, targ_feats, source_feats)
	for i = 1,inds:size(1) do
		local index = inds[i]
		targ_feats[index]:copy(source_feats[i]) --copy over the first self.feature_dim features
	end
end

function SRLBatcher:slow_scatter3(sparse,dense)
	for i = 1,sparse:size(1) do
		local row_index = sparse[i][1]
		local col_index = sparse[i][2]
		dense[row_index][col_index] = 1
	end
end

function SRLBatcher:get_iterator()
	self.cur_idx = 1
	local ended = false
	self:init_tracking_of_rejcted_examples()
	return function()
		if(ended) then return {self:report_on_rejected_examples(),nil,0} end
		local labs, feats, num, ended_this_batch = unpack(self:get_next_batch())
		ended = ended_this_batch
		return {labs, feats, num}
	end
end

function SRLBatcher:get_ongoing_iterator(shuffle)
	assert(not shuffle, 'not implemented')

	return function()
		local labs, feats, num, end_reached = unpack(self:get_next_batch())
		return {labs, feats, num}
	end
end

	--local flattened_features = dense_features:view(self.max_rows*self.max_cols,feature_dim
	--local linearize_indices = row_indices:clone():mul(self.max_cols) + col_indices
	--torch.scatter(dense_features,features,linearize_indices)