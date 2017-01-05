local Inference1DUtil = torch.class('Inference1DUtil')
package.path = package.path .. ';../?.lua'
require 'ReshapeAs'


--TODO: it's possible that there is a built-in function for this
function Inference1DUtil:softmax_3d(y_shape)
	local batch_size,length, domain_size = unpack(y_shape)
	local s = nn.Sequential()
	s:add(nn.View(batch_size*length,domain_size))
	s:add(nn.SoftMax())
	s:add(nn.View(unpack(y_shape)))

	return s
end

function Inference1DUtil:softmax_4d(y_shape)
	local batch_size,height, width, domain_size = unpack(y_shape)
	local s = nn.Sequential()
	s:add(nn.View(batch_size*height*width,domain_size))
	s:add(nn.SoftMax())
	s:add(nn.View(unpack(y_shape)))

	return s
end


function Inference1DUtil:expand_tile_flatten(t,y_shape,expand_across_row)
	local batch_size, length, domain_size = unpack(y_shape)
	t = nn.Replicate(domain_size,4,4)(t)
	if(not expand_across_row) then
		t = nn.Transpose({3,4})(t)
	end
	return nn.Reshape(batch_size,length-1,domain_size,domain_size,false)(t)
end

function Inference1DUtil:make_outer_product_marginals(y,y_shape)

	local length = y_shape[2]
	local y_left = nn.Narrow(2,1,length-1)(y)
	local y_right = nn.Narrow(2,2,length-1)(y)

	local y_left_tile = Inference1DUtil:expand_tile_flatten(y_left,y_shape,true)
	local y_right_tile = Inference1DUtil:expand_tile_flatten(y_right,y_shape,false)

	return nn.CMulTable()({y_left_tile,y_right_tile})
end

function Inference1DUtil:make_onehots(y,y_shape)
	local batch_size, length, domain_size = unpack(y_shape)

	local out = torch.zeros(unpack(y_shape))
	for b = 1,batch_size do
		for t = 1,length do
			local idx = y[b][t][1]
			out[b][t][idx] = 1
		end
	end
	return out
end

--this is a misnomer. it expects int values
function Inference1DUtil:make_outer_product_marginals_from_onehots(y,y_shape)
	local batch_size, length, domain_size = unpack(y_shape)

	local margs = torch.zeros(batch_size,length - 1, domain_size, domain_size)
	for b = 1,batch_size do
		for t = 1,(length-1) do
			local y_left_value = y[b][t][1]
			local y_right_value = y[b][t+1][1]
			margs[b][t][y_left_value][y_right_value] = 1
		end
	end
	return margs
end


function Inference1DUtil:edge_marginals_to_node_marginals(edge_marginals, y_shape)
	local batch_size, length, domain_size = unpack(y_shape)


	--input: b x l -1 x d x d
	-- b x l-1 x d
	local node_marginals_except_last = edge_marginals:sum(4):view(batch_size, length-1, domain_size)--TODO: is this correct?
	
	--b x d
	local final_node_marginal = edge_marginals:narrow(2,length-1, 1):reshape(batch_size, domain_size, domain_size):sum(2):reshape(batch_size,1,domain_size)
	local node_marginals = nn.JoinTable(2):forward({node_marginals_except_last,final_node_marginal})

	return node_marginals
end

function Inference1DUtil:assert_calibrated_edge_marginals(edge_marginals, y_shape)
	local batch_size, length, domain_size = unpack(y_shape)

	if(length > 1) then
		for i =1,(length-2) do
			local pairwise_margs_from_left  = edge_marginals:narrow(2,i,1):reshape(torch.LongStorage({batch_size, domain_size, domain_size}))
			local pairwise_margs_from_right = edge_marginals:narrow(2,i+1,1):reshape(torch.LongStorage({batch_size, domain_size, domain_size}))

			--First, check that the clique marginals sum to one
			assert(pairwise_margs_from_left:sum(3):sum(2):add(-1):norm() < 0.00001)
			
			--These are both estimates of the node marginals at timestep i+1
			local margs_from_left = pairwise_margs_from_left:sum(2):view(torch.LongStorage({batch_size,domain_size}))
			local margs_from_right = pairwise_margs_from_right:sum(3):view(torch.LongStorage({batch_size,domain_size}))
			
			--Check that each node marginal sums to one.
			assert(margs_from_left:sum(2):add(-1):norm() < 0.00001)
			assert(margs_from_right:sum(2):add(-1):norm() < 0.00001)

			--Check that they agree.
			assert((margs_from_left - margs_from_right):norm() < 0.0001)
		end
	end
end
