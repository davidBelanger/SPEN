package.path = package.path .. ';../torch-util/?.lua'
require 'Util'

local toks = arg[1]
local labels = arg[2]


local all_toks = {}
local delim = " "
for line in io.lines(toks) do
	local d = Util:splitByDelim(line,delim)
	table.insert(all_toks,d)
end

local all_labels = {}
for line in io.lines(labels) do
	local d = Util:splitByDelim(line,delim)
	table.insert(all_labels,d)
end

local len = #all_toks[1]

local toks_tensor = torch.Tensor(#all_toks,len)
for i = 1,#all_toks do
	for j = 1,len do
		toks_tensor[i][j] = all_toks[i][j]
	end
end

local len = #all_labels[1]
local labels_tensor = torch.Tensor(#all_toks,len)
for i = 1,#all_toks do
	for j = 1,len do
		labels_tensor[i][j] = all_labels[i][j]
	end
end

print('toks size')
print(toks_tensor:size())
print('labels size')
print(labels_tensor:size())
local data = 
{
	labels = labels_tensor,
	data = toks_tensor
}

torch.save(arg[3],data)
