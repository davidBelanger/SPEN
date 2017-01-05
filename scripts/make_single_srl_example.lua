local a  = torch.load('srl/processed_data/conll2005/train.arcs.torch')

local f  = torch.load('srl/processed_data/conll2005/train.features.torch')


local arcs = {}
local count = 0
local m1 = -1
local m2 = -1
for i = 1,32 do
	table.insert(arcs,a[i])
	count = count + a[i]:size(1)
end

local ff = f:narrow(1,1,count):clone()
torch.save('srl/processed_data/conll2005/train.arcs.torch.small',arcs)
torch.save('srl/processed_data/conll2005/train.features.torch.small',ff)


