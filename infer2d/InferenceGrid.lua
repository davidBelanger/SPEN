local InferenceGrid = torch.class('InferenceGrid')
package.path = package.path .. ';../?.lua'
require 'ReshapeAs'

function InferenceGrid:__init(y_shape)
	self.y_shape = y_shape
	self.batch_size, self.height, self.width, self.domain_size = unpack(y_shape)
end

--TODO: it's possible that there is a built-in function for this
function InferenceGrid:softmax_4d()
	local s = nn.Sequential()
	s:add(nn.View(self.batch_size*self.height*self.width,self.domain_size))
	s:add(nn.SoftMax())
	s:add(nn.View(unpack(self.y_shape)))

	return s
end

-- --TODO: this stuff is only necessary if doing GD inference

-- function InferenceGrid:expand_tile_flatten(t,expand_across_row)
-- 	t = nn.Replicate(self.domain_size,4,4)(t)
-- 	if(not expand_across_row) then
-- 		t = nn.Transpose({3,4})(t)
-- 	end
-- 	return nn.Reshape(self.batch_size,self.length-1,self.domain_size,self.domain_size,false)(t)
-- end

-- function InferenceGrid:make_outer_product_marginals(y)
-- 	local y_left = nn.Narrow(2,1,self.length-1)(y)
-- 	local y_right = nn.Narrow(2,2,self.length-1)(y)

-- 	local y_left_tile = self:expand_tile_flatten(y_left,true)
-- 	local y_right_tile = self:expand_tile_flatten(y_right,false)

-- 	return nn.CMulTable()({y_left_tile,y_right_tile})
-- end

-- function InferenceGrid:make_outer_product_marginals_from_onehots(y)
-- 	local margs = torch.zeros(self.batch_size,self.length - 1, self.domain_size, self.domain_size)
-- 	for b = 1,self.batch_size do
-- 		for t = 1,(self.length-1) do
-- 			local y_left_value = y[b][t][1]
-- 			local y_right_value = y[b][t+1][1]
-- 			margs[b][t][y_left_value][y_right_value] = 1
-- 		end
-- 	end
-- 	return margs
-- end


-- function InferenceGrid:edge_marginals_to_node_marginals(edge_marginals)
-- 	--input: b x l -1 x d x d
-- 	-- b x l-1 x d
-- 	local node_marginals_except_last = edge_marginals:sum(4):view(self.batch_size, self.length-1, self.domain_size)--TODO: is this correct?
	
-- 	--b x d
-- 	local final_node_marginal = edge_marginals:narrow(2,self.length-1, 1):reshape(self.batch_size, self.domain_size, self.domain_size):sum(2):reshape(self.batch_size,1,self.domain_size)
-- 	local node_marginals = nn.JoinTable(2):forward({node_marginals_except_last,final_node_marginal})

-- 	return node_marginals
-- end

-- function InferenceGrid:assert_calibrated_edge_marginals(edge_marginals)

-- 	if(self.length > 1) then
-- 		for i =1,(self.length-2) do
-- 			local pairwise_margs_from_left  = edge_marginals:narrow(2,i,1):reshape(torch.LongStorage({self.batch_size, self.domain_size, self.domain_size}))
-- 			local pairwise_margs_from_right = edge_marginals:narrow(2,i+1,1):reshape(torch.LongStorage({self.batch_size, self.domain_size, self.domain_size}))

-- 			--First, check that the clique marginals sum to one
-- 			assert(pairwise_margs_from_left:sum(3):sum(2):add(-1):norm() < 0.00001)
			
-- 			--These are both estimates of the node marginals at timestep i+1
-- 			local margs_from_left = pairwise_margs_from_left:sum(2):view(torch.LongStorage({self.batch_size,self.domain_size}))
-- 			local margs_from_right = pairwise_margs_from_right:sum(3):view(torch.LongStorage({self.batch_size,self.domain_size}))
			
-- 			--Check that each node marginal sums to one.
-- 			assert(margs_from_left:sum(2):add(-1):norm() < 0.00001)
-- 			assert(margs_from_right:sum(2):add(-1):norm() < 0.00001)

-- 			--Check that they agree.
-- 			assert((margs_from_left - margs_from_right):norm() < 0.0001)
-- 		end
-- 	end
-- end
