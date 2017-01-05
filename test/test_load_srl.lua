package.path = package.path .. ';batch/?.lua'
require 'SRLBatcher'

require 'nn'

local labels_file = "srl/processed_data/conll2005/dev.arcs.torch"
local features_file = "srl/processed_data/conll2005/dev.features.torch"
local max_rows = 100 --max number of predicates
local max_cols = 100 --max number of arguments
local null_arc_index = 0
local batch_size = 32
local feature_dim = 128

local batcher = SRLBatcher(labels_file,features_file,batch_size, feature_dim, max_rows,max_cols, null_arc_index)

local iter = batcher:get_iterator()

local data = iter()
local total = 0
while(data[2]) do
	total = total + data[3]
	data = iter()
end

print(total)
local a = torch.load('srl/processed_data/conll2005/dev.arcs.torch')
print(#a)

assert(#a == total)


