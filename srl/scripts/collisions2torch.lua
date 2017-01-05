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
local linecount = 0
for line in io.lines(input) do
	linecount = linecount + 1

	local dat = split_by_delim(line," ",false)
	
	local int_data = torch.ByteTensor(#dat,2):fill(0)
	
	collision_pairs = {}
	for i = 1,#dat do
		k1, k2 = unpack(split_by_delim(dat[i],",",true))
		int_data[i][1] = k1 + 1 --add one because lua is 1-indexed
		int_data[i][2] = k2 + 1
	end
	
	if(linecount % 500 == 0) then print(linecount) end
	
	table.insert(tab,int_data)
end

print('processed # lines: '..linecount)
torch.save(output,tab)
