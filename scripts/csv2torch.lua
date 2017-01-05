local input = arg[1]
local output = arg[2]
require 'nn'

function split_by_delim(str,delim,convertFromString)
	local convertFromString = convertFromString or false

	local function convert(input)
		if(convertFromString) then  return tonumber(input)  else return input end
	end

    local t = {}
    local pattern = '([^'..delim..']+)'
    for word in string.gmatch(str, pattern) do
     	table.insert(t,convert(word))
    end
    return t
end

local tab = {}

local c = 0
for line in io.lines(input) do
	local dat = split_by_delim(line," ",true)
	dat = torch.Tensor(dat)
	dat:resize(1,dat:size(1))
	table.insert(tab,dat)
	c = c+1
	if(c % 2000 == 0) then print(c) end
end

local all_data = nn.JoinTable(1):forward(tab)

torch.save(output,all_data)

