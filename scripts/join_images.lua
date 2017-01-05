require 'nn'
local file_list = arg[1]
local out = arg[2]

local data = {}
local labels = {}
local count = 0
for line in io.lines(file_list) do
    local a = torch.load(line)
    table.insert(data,a.data)
    table.insert(labels,a.labels)
    count = count + a.data:size(1)
end

local data_all = nn.JoinTable(1):forward(data)
local labels_all = nn.JoinTable(1):forward(labels)
assert(count == data_all:size())

local tab = {labels = labels_all, data = data_all}
torch.save(out,tab)
