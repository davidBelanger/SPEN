local MultiLabelEvaluation = torch.class('MultiLabelEvaluation')

function MultiLabelEvaluation:__init(batcher,predictor,predictionThresh,resultsFile)
	self.batcher = batcher
	self.predictor = predictor
	self.predictionThreshes = (predictionThresh > 0) and predictionThresh or {0,0.01, 0.02, 0.03, 0.04, 0.05,0.10,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.60,0.65,0.70,0.75}

	self.resultsFile = resultsFile

end

function MultiLabelEvaluation:evaluate(iterName)
	print('STARTING EVALUATION')
	self.predictor:evaluate()

	local pits = {}
	local gits = {}
	
	local exampleCount = 0
	local batch_iterator = self.batcher:get_iterator()
	local batch_count = 0
	while(true) do
		batch_count = batch_count + 1
		local batch_data = batch_iterator()
		if(batch_data == nil) then break end

		local batch_labels, batch_inputs, num_actual_data = unpack(batch_data)

		local preds = self.predictor:forward(batch_inputs):select(3,2)
		local pit = preds:narrow(1,1,num_actual_data)

		local git = batch_labels:narrow(1,1,num_actual_data)
		table.insert(pits,pit:clone())
		table.insert(gits,git:clone():add(-1))	--this assumes that the loaded data has 1-2 indexing, whereas the logic below uses 0-1 indexing, so we subtract 1 from the labels
	
		exampleCount = exampleCount + num_actual_data
	end

	local best_f1_ex = -1
	local best_f1_lab = -1

	local pit_all = nn.JoinTable(1):forward(pits)
	local git_all = nn.JoinTable(1):forward(gits)
	local bestThresh = -1
	-- local bestHamming = 5
	-- local bestHammingThresh = 5

	-- local sortedScores	
	-- if(self.predictionType == "globalThresh") then
	-- 	sortedScores = pit_all:reshape(pit_all:nElement()):sort(true)
	-- end
	print('True positive rate: '..git_all:float():sum()/git_all:nElement())

	for _,thresh in ipairs(self.predictionThreshes) do
		local pi = pit_all:clone()
		local gi = git_all:clone():double()

		pi = pi:ge(thresh):double()

		local correct = torch.cmul(pi,gi)

		print('Thresh = '..thresh)
		print('num predictions = '..pi:sum())

		print('Label-Averaged: ')
		local f1_label,prec_label,rec_label = self:F1(correct,pi,gi,1)
		best_f1_lab = math.max(f1_label,best_f1_lab)
		print(string.format('F1: %f, Prec: %f, Rec: %f',f1_label,prec_label,rec_label))

		print('Example-Averaged: ')
		local f1_ex,prec_ex,rec_ex = self:F1(correct,pi,gi,2)
		if(best_f1_ex < f1_ex) then 
			best_f1_ex = f1_ex
			bestThresh = thresh 
		end
		print(string.format('F1: %f, Prec: %f, Rec: %f',f1_ex,prec_ex,rec_ex))

		-- local missed = pi:add(gi):eq(1.0)
		-- local hamming = missed:sum()/missed:nElement()
		-- print('Hamming: '..hamming)
		-- if(hamming < bestHamming)then  bestHammingThresh = thresh end
		-- bestHamming = math.min(hamming,bestHamming)

	end 

	print('BEST LAB = '..best_f1_lab)
	print('BEST thresh = '..bestThresh)
	print('BEST EX = '..best_f1_ex)
	--print('BEST Hamming = '..bestHamming)
	--print('BEST Hamming thresh = '..bestHammingThresh)
	
	self.predictor:training()
	
	print('computed on '..exampleCount.." examples")
	print('')
end

--this expects 0-1 matrices as input
--it computes either micro-averaged or macro-averaged F1, where averaging is done over dimension dim
function MultiLabelEvaluation:F1(correct,predicted,gt,dim)
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

