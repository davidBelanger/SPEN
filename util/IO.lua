local IO = torch.class('IO')

function IO:save_batcher(batcher, num, filename)

	local batch_labels = {}
	local batch_data = {}
	local iter = batcher:get_iterator()
	for t = 1,num do
		local labs, data = unpack(iter())
		table.insert(batch_labels, labs)
		table.insert(batch_data, data)
	end

	local labels = nn.JoinTable(1):forward(batch_labels)
	local data = nn.JoinTable(1):forward(batch_data)
	torch.save(filename,{labels = labels, data = data})
end

function IO:save_batcher_to_csv(batcher, num, label_file, feature_file)

	local batch_labels = {}
	local batch_data = {}
	local iter = batcher:get_iterator()
	for t = 1,num do
		local labs, data = unpack(iter())
		table.insert(batch_labels, labs)
		table.insert(batch_data, data)
	end

	local labels = nn.JoinTable(1):forward(batch_labels)
	local data = nn.JoinTable(1):forward(batch_data)
	IO:save_csv(label_file,labels)
	IO:save_csv(feature_file,data)
end

function IO:save_csv(out_path, data)
	assert(data:dim() == 2)

	-- Opens a file in append mode
	local file = io.open(out_path, "w")

	for i = 1,data:size(1) do
		for j = 1,data:size(2) do
			file:write(data[i][j])
			if(j < data:size(2)) then file:write(',') end
		end
		file:write('\n') 
	end
	file:close()
end


function IO:load_csv(filename)
	return IO:load_ascii(filename,',')
end

--TODO: this isn't particularly optimized code
function IO:load_ascii(filename,delimeter)
	local data = {}
	li = 0
	for line in io.lines(filename) do
		local tab = IO:splitByDelim(line,delimeter,true)
		local d = torch.Tensor(tab)
		d = d:view(1,d:size(1))
		table.insert(data,d)
	end
	require 'nn'
	local merged_data = nn.JoinTable(1):forward(data)
	return merged_data
end

function IO:splitByDelim(str,delim,convertFromString)
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