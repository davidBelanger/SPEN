local PSNREvaluation = torch.class('PSNREvaluation')

function PSNREvaluation:__init(resultsFile,writeIms)
	self.resultsFile = resultsFile 
	self.writeIms = writeIms
end

function PSNREvaluation:evaluateClassifier(batcher,net,iterName)
	return self:evaluate(batcher,net,iterName)
end

--todo: add better logging
function PSNREvaluation:evaluate(batcher,net,name)
	local count = 0
	local total_correct = 0
	batcher:reset()
	print('STARTING EVALUATION')
	net:evaluate()

	local errSum = 0
    local exampleCount = 0

    local outDir = self.resultsFile.."/"..name.."/"
    if(self.writeIms) then 
    	print('writing output images to '..outDir)
    	os.execute('mkdir -p '..outDir) 
    end

    local cnt = 0
	while(true) do
		local batch_labels, batch_inputs, num_actual_data = batcher:getBatch()
		if(batch_inputs == nil) then break end

		local preds = net:forward(batch_inputs)

		if(not torch.isTensor(preds)) then preds = preds[#preds] end --for compatibility with other things that return a whole time series of iterates

		if(self.writeIms) then 
			for j = 1,batch_labels:size(1) do
				cnt = cnt + 1
			    image.save(outDir..cnt.."-input.jpg", batch_inputs[j])
			    image.save(outDir..cnt.."-pred.jpg", preds[j])
			    image.save(outDir..cnt.."-gt.jpg", batch_labels[j])
			end
		end


		local pp = preds:narrow(1,1,num_actual_data)
		local tt = batch_labels:narrow(1,1,num_actual_data)
		if(not self.mse) then
			self.mse = nn.MSECriterion()
			if(preds:type() == 'torch.CudaTensor') then self.mse:cuda() end
		end
		local err = self.mse:forward(pp,tt)
		errSum = errSum + err*num_actual_data
		exampleCount = exampleCount + num_actual_data
		
    end
    local avgMSE = errSum/exampleCount
    --note that this assumes that the images are in [0,1]. Otherwise, we'd have something like 255^2 in the numerator.
    local psnr = 10*torch.log(1/avgMSE)/torch.log(10)
	print('PSNR: '..psnr)
	print('')
	net:training()

end


