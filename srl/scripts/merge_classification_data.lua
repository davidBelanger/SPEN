

local label_file = arg[1]
local data_file = arg[2]
local out = arg[3]

--first, read all of the labels into a tensor
local labels = {}
for l in io.lines(label_file) do
	table.insert(labels,tonumber(l))
end

labels = torch.Tensor(labels)
labels:add(1) --because torch is 1-indexed

local feature_dim = 128
local num_labels = labels:size(1)
local data = torch.Tensor(num_labels, feature_dim)

local i = 0  
for line in io.lines(data_file) do  
  i = i + 1
  local l = line:split(' ')
  for key, val in ipairs(l) do
    data[i][key] = val
  end
  if(i % 2500 == 0) then print(i*100.0/num_labels) end
end


torch.save(out,{data=data,labels=labels})
