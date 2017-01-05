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
local joiner = nn.JoinTable(1)
local c = 0
local cur_sentence = {}
local cur_sent_id = nil
local linecount = 0
for line in io.lines(input) do
	linecount = linecount + 1

	local dat = split_by_delim(line," ",true)
	table.insert(dat,linecount)

	local sent_id = dat[1]
	cur_sent_id = (linecount == 1) and sent_id or cur_sent_id

	dat = torch.IntTensor(dat)
	dat:narrow(1,2,3):add(1) --make the indexing and labels 1-indexed
	dat:resize(1,dat:size(1))

	if(sent_id ~= cur_sent_id) then
		local joined = joiner:forward(cur_sentence):clone()
		table.insert(tab,joined)
		cur_sentence = {}
		cur_sent_id = sent_id
		c = c + 1
		if(c % 500 == 0) then print(c) end
	end
	table.insert(cur_sentence,dat)
end
local joined = joiner:forward(cur_sentence):clone()
table.insert(tab,joined)
print(#tab.." entries")
torch.save(output,tab)
