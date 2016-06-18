local MultiLabelEvaluation = torch.class('MultiLabelEvaluation')

function MultiLabelEvaluation:__init(predictionThreshes,predictionType,analyzer,resultsFile)
	self.predictionThreshes = predictionThreshes
	self.predictionType =  predictionType or false
	self.analyzer = analyzer
	self.resultsFile = resultsFile
end

function MultiLabelEvaluation:evaluateClassifier(batcher,net,iterName)
	return self:evaluate(batcher,net,true,false,iterName)
end

function MultiLabelEvaluation:evaluateInference(batcher,net,initAtPred,iterName)
	return self:evaluate(batcher,net,useClassifier,initAtPred,iterName)
end

function MultiLabelEvaluation:evaluate(batcher,net,useClassifier,initAtPred,iterName)
	local exampleCount = 0
	batcher:reset()
	print('STARTING EVALUATION')
	net:evaluate()
	local initPoint = nil
	if(initAtPred) then initPoint = "pred" end

	local count = 0
	local pits = {}
	local gits = {}
	local totalInfo = {}
	local function mean(tab)
		return torch.Tensor(tab):mean()
	end
	local function sum(tab)
		return torch.Tensor(tab):sum()
	end

	local function accumulateInfo(info)
		if(useClassifier) then
			if(not totalInfo.totalTime) then 
				totalInfo.totalTime  = 0 
				totalInfo.count = 0
			end
			totalInfo.totalTime = totalInfo.totalTime + info.time
			totalInfo.count = totalInfo.count + 1
		else
			if(info.totalTime) then
				totalInfo.cheatingTimes = totalInfo.cheatingTimes or {}
				totalInfo.totalTimes = totalInfo.totalTimes or {}
				totalInfo.numIters = totalInfo.numIters or {}
				local convergenceIters = info.iterateConvergenceIters:cmin(info.objConvergenceIters) --todo: change back to cmin
				if(not totalInfo.convergenceIters) then
					totalInfo.convergenceIters =  convergenceIters
				else
					totalInfo.convergenceIters = torch.cat(totalInfo.convergenceIters,convergenceIters)
				end

				local cheatingTime = info.elapsedFeatureTime + (convergenceIters:mean()/info.necessaryIters)*info.inferenceTime
				table.insert(totalInfo.cheatingTimes, cheatingTime)
				table.insert(totalInfo.totalTimes,info.totalTime)
				table.insert(totalInfo.numIters,info.necessaryIters)
			end
		end
	end

	local function finalizeInfo()
		if(useClassifier) then
			print('total time = '..totalInfo.totalTime)
		else
			if(totalInfo.convergenceIters) then
				print('avg iters to per-example convergence: '..mean(totalInfo.convergenceIters))
				if(self.resultsFile ~= "") then
					assert(iterName)
					local ff = self.resultsFile.."-"..iterName
					print('writing to '..ff)
					local out = io.open(ff,'w')
					for i = 1,totalInfo.convergenceIters:size(1) do
						out:write(totalInfo.convergenceIters[i].."\n")
					end
					out:close()
				end
				print(Histogram(totalInfo.convergenceIters,15))
				print('avg iters to per-block convergence: '..mean(totalInfo.numIters))
			end

			if(totalInfo.totalTimes) then
				print('avg total time per block = '..mean(totalInfo.totalTimes))
				print('total time = '..sum(totalInfo.totalTimes))
			end

				--todo: also plot necessary iters
			if(totalInfo.cheatingTimes) then
				print('avg cheating time = '..mean(totalInfo.cheatingTimes))
				print('total cheating time = '..sum(totalInfo.cheatingTimes))

			end
		end
	end
	searchErrorSum = 0
	local computeSearchErrors = false--not useClassifier
	
	while(true) do
		local batch_labels, batch_inputs, num_actual_data = batcher:getBatch()
		assert(count == 0,'currently this is set up to only handle a single input file')
		if(batch_inputs == nil) then break end
		local pit = nil
		local git = nil

		if(useClassifier) then
			local start = os.clock()
			local numExtraTrials = 0-- use something nonzero and big,like 100, when trying to get timing numbers
			for i = 1,numExtraTrials do
				net:forward(batch_inputs)
			end
			if(not torch.isTensor(batch_inputs)) then print(batch_inputs) end
			local preds = net:forward(batch_inputs)
			local elapsedTime = (os.clock() - start )/(numExtraTrials+1)
			pit = preds:narrow(1,1,num_actual_data)
			git = batch_labels:narrow(1,1,num_actual_data)
			table.insert(pits,pit:clone())
			table.insert(gits,git:clone())		
			accumulateInfo({time = elapsedTime})	
		else 
			batch_labels = batch_inputs[1]
			local inferencer = net
			local inferred_score, inferred_labels, numIters, info = inferencer:doInference(batch_inputs,nil,false,initPoint)
			local ground_truth_score
			local search_errors = 0

			accumulateInfo(info)

			if(computeSearchErrors) then
				ground_truth_score = inferencer:forwardOnLabels(batch_inputs[1])
				search_errors = ground_truth_score:add(-1,inferred_score):cdiv(inferred_score:clone():abs()):lt(-0.001):double()
				searchErrorSum = searchErrorSum + search_errors:sum()
				print('search error rate = '..search_errors:mean())
				 --compute lt bc inference does minimization
			end
			local peak = inferencer:peakedness(inferred_labels)
			print('-----------')
			print(num_actual_data)
						print('ooooooooooo')

			print(batch_labels:size())
			local il = inferred_labels:narrow(1,1,num_actual_data)
			local bl = batch_labels:narrow(1,1,num_actual_data)
			print('finished batch. required # iters = '..numIters.." peak = "..peak)
			pit = il
			git = bl
			--this way, we can handle things that explicitly have a positive and negative dimension
			if(il:dim() == 3) then
				pit = il:narrow(3,2,1):squeeze() --this slices out the 'positive' label
				git = bl:narrow(3,2,1):squeeze()
			end
			table.insert(pits,pit:clone())
			table.insert(gits,git:clone())
		end

		exampleCount = exampleCount + num_actual_data
		if(computeSearchErrors) then
			print('Total search error rate = '..(searchErrorSum/exampleCount))
		end
	end

	local best_f1_ex = -1
	local best_f1_lab = -1

	local pit_all = nn.JoinTable(1):forward(pits)
	local git_all = nn.JoinTable(1):forward(gits)
	local bestThresh = -1
	local bestHamming = 5
	local bestHammingThresh = 5

	local sortedScores	
	if(self.predictionType == "globalThresh") then
		sortedScores = pit_all:reshape(pit_all:nElement()):sort(true)
	end

	local cnt = 0
	for _,thresh in ipairs(self.predictionThreshes) do
		cnt = cnt + 1
		local pi = pit_all:clone()
		local gi = git_all:clone():double()

		if(not self.predictionType) then
			pi = pi:ge(thresh):double()
		elseif(self.predictionType == "globalThresh") then
			idx = math.ceil(sortedScores:size(1)*(cnt/#self.predictionThreshes))
			print(idx.." out of "..sortedScores:size(1))
			thresh = sortedScores[idx]
			pi = pi:ge(thresh):double()
		elseif(self.predictionType == "rowWise") then
	    	assert(pi:dim() == 2)
	    	local m, mi = pi:sort(2,true)
	    	pi = pi:ge(m:narrow(2,cnt,1):expandAs(pi)):double()
	    	local sums = pi:sum(2)
	    	--[[ this condition should hold, but it gets messed up by ties
	    	local cond = sums:eq(cnt):all()
	    	if(not cond) then
	    		print(pi:min())
	    		print(pi:max())
	    		print(sums:min())
	    		print(sums:max())
	    	end
	    	assert(cond)
	    	--]]

	    else
	    	os.error("shouldn't be here")
	    end

		local correct = torch.cmul(pi,gi)
		local wrongPredictions = torch.gt(pi,gi)
		print('Thresh = '..thresh)
		print('num predictions = '..pi:sum())
		print('wrong predictions = '..wrongPredictions:sum())

		print('Label-Averaged: ')
		local f1_label,prec_label,rec_label = self:microF1(correct,pi,gi,1)
		best_f1_lab = math.max(f1_label,best_f1_lab)
		print(string.format('F1: %f, Prec: %f, Rec: %f',f1_label,prec_label,rec_label))

		print('Example-Averaged: ')
		local f1_ex,prec_ex,rec_ex = self:microF1(correct,pi,gi,2)
		if(best_f1_ex < f1_ex) then bestThresh = thresh end
		best_f1_ex = math.max(f1_ex,best_f1_ex)
		print(string.format('F1: %f, Prec: %f, Rec: %f',f1_ex,prec_ex,rec_ex))

		local missed = pi:add(gi):eq(1.0)
		local hamming = missed:sum()/missed:nElement()
		print('Hamming: '..hamming)
		if(hamming < bestHamming)then  bestHammingThresh = thresh end
		bestHamming = math.min(hamming,bestHamming)

	end 

	print('BEST LAB = '..best_f1_lab)
	print('BEST thresh = '..bestThresh)
	print('BEST EX = '..best_f1_ex)
	print('BEST Hamming = '..bestHamming)
	print('BEST Hamming thresh = '..bestHammingThresh)
	
	if(self.analyzer) then
		local pi = pit_all:ge(bestThresh):double()
		self.analyzer(pi)
	end

	net:training()
	
	print('computed on '..exampleCount.." examples")
	print('')
	finalizeInfo()
end

--this expects 0-1 matrices as input
--it computes micro-averaged f1, where averaging is done over dimension dim
function MultiLabelEvaluation:microF1(correct,predicted,gt,dim)
	local eps =  0.0000001
	local correct_sums = correct:sum(dim):squeeze()
	local predicted_sums = predicted:sum(dim):squeeze()
	local gt_sums = gt:sum(dim):squeeze()
	correct_sums:add(eps)
	predicted_sums:add(eps)
	gt_sums:add(eps)

	local precisions = torch.cdiv(correct_sums,predicted_sums)
	local recalls = torch.cdiv(correct_sums,gt_sums)

	local f1s = torch.cmul(precisions,recalls):mul(2)
	f1s:cdiv(precisions + recalls)
	return f1s:mean(),precisions:mean(), recalls:mean()
end

--this expects numbers as input
function MultiLabelEvaluation:f1(correct,predicted,gt)
	local precision = correct/predicted
	local recall = correct/gt
	local f1 = 2*precision*recall/(precision + recall)
	return f1,precision,recall
end

