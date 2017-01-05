require 'nn'
require 'nngraph'

local size1 = {5,6}
local size2 = {5,7}
local a = torch.randn(unpack(size1))
local b = torch.randn(unpack(size2))

function outer_product(size1,size2)
	local u = nn.Identity()()
	local v = nn.Identity()()
	local u1 = nn.Reshape(size1[1], size1[2], 1, false)(u)
	local v1 = nn.Reshape(size2[1], size2[2], 1, false)(v)
	local prod = nn.MM(false,true)({u1,v1})
	return nn.gModule({u,v},{prod})
end

local net = outer_product(size1,size2)


local prod = net:forward({a,b})

for k = 1,size1[1] do
	for i = 1,size1[2] do
		for j = 1,size2[2] do
			assert(prod[k][i][j] == a[k][i]*b[k][j])
		end
	end
end

