local gm = require 'graphicsmagick'
package.path = package.path .. ';../../torch-util/?.lua'
require 'Util'
require 'xlua'

local fileList = arg[1]
local numPerFile = arg[2]*1
local outputFile = arg[3]
local numTotal = arg[4]*1

local fileCount = 0
local lineCount = 0
local outputTensor

local function save()
	local fn = outputFile.."-"..fileCount..'.t7'
	print('writing '..fn)
	torch.save(fn,outputTensor)
	fileCount = fileCount + 1
end

numChannels = 1
local function initData(e_targets,e_inputs) 
	local targets = torch.Tensor(numPerFile,numChannels,e_targets:size(1),e_targets:size(2))
	local inputs = torch.Tensor(numPerFile,numChannels,e_inputs:size(1),e_inputs:size(2))

	local dat = {targets,inputs}
	return dat
end


local function load_image(path)
	local i = gm.Image(path)
	return i:toTensor('float','I'):select(3,1) --this is because they are assumed to be grayscale
end
local function copyData(bigData,gt,blur,i) 
	bigData[1][i]:copy(gt)
	bigData[2][i]:copy(blur)
end

local data = {}
xlua.progress(0,numTotal)
local totalProcessed = 0
for line in io.lines(fileList) do
	if(lineCount % numPerFile == 0 and lineCount > 0) then
		save()
		outputTensor = nil
		lineCount = 0
	end
	totalProcessed = totalProcessed + 1

	local fields = Util:splitByDelim(line,' ',false)
	local im_blur   = load_image(fields[1])
	local im_gt = load_image(fields[2])

	lineCount = lineCount + 1

	if(not outputTensor) then
		outputTensor = initData(im_gt,im_blur)
	end
	copyData(outputTensor,im_gt,im_blur,lineCount)
	xlua.progress(totalProcessed,numTotal)
end

save()
