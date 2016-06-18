local ImageAnalysis = torch.class('ImageAnalysis')


function ImageAnalysis:__init(dir,numExamples)
	self.dir = dir
	self.numExamples = numExamples
end


function ImageAnalysis:makeExamples(testBatcher,net,i)
   	local targets,inputs = testBatcher:getBatch()

   	local output = net:forward(inputs)
   	if(not torch.isTensor(output)) then output = output[#output] end --for compatibility with other things that return a whole time series of iterates

	for j = 1,math.min(self.numExamples,output:size(1)) do
	    image.save(self.dir..'/iter-'..i.."-"..j.."-input.jpg", inputs[j])
	    image.save(self.dir..'/iter-'..i.."-"..j.."-pred.jpg", output[j])
	    image.save(self.dir..'/iter-'..i.."-"..j.."-gt.jpg", targets[j])
	end
end




