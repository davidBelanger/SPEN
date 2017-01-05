require 'Imports'

local gt = torch.randn(5,4)
local num_tiles = 6
local pred = torch.randn(num_tiles,5,4)

local pred_split = nn.SplitTable(1):forward(pred)
local weights = torch.rand(num_tiles)
local crit = nn.RepeatedCriterion(nn.MSECriterion(),weights)

local loss1 = crit:forward(pred_split,gt)


local loss2 = 0
for i = 1,num_tiles do
	loss2 = loss2 + (1/num_tiles)*weights[i]*nn.MSECriterion():forward(pred[i],gt)
end

assert(loss2 == loss1)

local bg = crit:backward(pred_split,gt)

