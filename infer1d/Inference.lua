local Inference = torch.class('Inference')

--note: this does energy *mininimzation*
function Inference:__init(inference_net,features_net,predictor_net,cuda,minibatchsize,structured_training_loss,options,projector)
	self.inference_net = inference_net
	self.features_net = features_net
	self.structured_training_loss = structured_training_loss
	self.cuda = cuda
	self.assert = false
	self.loggingFreq = options.loggingFreq
	self.verbose = options.verbosity >1
	self.kinda_verbose = options.verbosity >=1  
	self.numIters = options.numIters
	self.objTol = options.objectiveConvergenceTolerance
	self.peakednessThresh = options.peakednessThresh
	self.minIters = options.minIters
	self.checkForPeak = options.checkForPeak
	self.iterateTol = options.iterateTol
	self.interpWeight = options.initInterpWeight
	self.predictor_net = predictor_net
	self.useL1 = options.labelL1  > 0
	self.l1 = options.labelL1
	self.computeTiming = options.analyzeInferenceTiming == 1
	self.singleProbability = options.singleProbability
	self.inferenceGradientClip = options.inferenceGradientClip
	self.clipGradient = self.inferenceGradientClip ~= 0
	self.usePerturbation = options.inferencePerturbation > 0
	self.inferencePerturbation = options.inferencePerturbation


	self.usePercentileForConvergence = options.convergencePercentile ~= 1.0
	self.convergencePercentile = options.convergencePercentile
	self.computePerExampleEnergy = self.usePercentileForConvergence

	self.trackRowConvergenceOverall = options.trackPerExampleConvergence == 1
	self.trackRowConvergence = true --this gets modified by training() and evaluate()

	local eW = options.entropyWeight*1
	if(options.clonePredictor) then
		self.predictor_net = predictor_net:clone()
	end
	self.projector = projector

	self.renormGradients = options.renormGradients or false

	self.changingBatchSize = true
	if(not (options.changingBatchSize == nil)) then
		self.changingBatchSize = options.changingBatchSize
	end

	if(self.singleProbability) then
		self.optimMethod = optim.projectedGradient
		self.optConfig = {
			learningRate = options.learningRate,
			momentum = options.momentum,
			dampening = 0,
			nesterov = options.nesterov,
			learningRateDecay = options.learningRateDecay,
			learningRatePower = options.learningRatePower,
			minValue = 0,
			maxValue = 1
		}
	else
		self.optimMethod = optim.emd 
		self.optConfig = {
			learningRate = options.learningRate,
			momentum = options.momentum,
			dampening = 0,
			nesterov = true, 
			checkNans = true,
			learningRateDecay = options.learningRateDecay,
			extraEntropyWeight = eW,
			learningRatePower = options.learningRatePower
		}
	end

	self.fixedUnitGradient = torch.Tensor(minibatchsize):fill(1.0)

	if(cuda) then 
		self.fixedUnitGradient = self.fixedUnitGradient:cuda() 
	end
end

function Inference:training()
	self.trackRowConvergence = false
	for _,net in ipairs({self.features_net,self.inference_net}) do 
		net:training()
	end
end

function Inference:evaluate()
	self.trackRowConvergence = self.trackRowConvergenceOverall
	for _,net in ipairs({self.features_net,self.inference_net}) do 
		net:evaluate()
	end
end

--this assumes that we pass it inputs = {labels,input_features}, though labels might be a dummy thing (we ignore its value anyway)
--if target_labels is nil, then we don't do loss-augmented inference
--Note that the reported obective in the inner loop of inference aggregates over the minibatch, whereas this function returns a per-example loss
function Inference:doInference(inputs,target_labels,loss_augmented,initPoint,clampTrueLabels)
		clampTrueLabels = clampTrueLabels or false --when this option is used, it only does inference over the labels which have a value of 0 in the ground truth (this is somewhat specific to multi-label classification)
		if(self.computeTiming) then
			self.startTime = sys.clock()
		end
		loss_augmented = loss_augmented or false
		assert(not (target_labels == nil and loss_augmented))

		local inference_labels = nil
		 --allocate a new one so that we don't have to worry about multiple calls to this yielding the same pointer
		 --TODO: this allocation is probably avoidable because performInference copies in the last step
		 --TODO: could also have the whole method run in copy mode or 'in place' mode to avoid allocations entirely
		inference_labels = inputs[1]:clone():zero()

		--initialize randomly
		local flatval = 1/inference_labels:size(inference_labels:dim())

		if(initPoint == "gt") then
	 		inference_labels:copy(target_labels):mul(1 - self.interpWeight):add(self.interpWeight*flatval)
			--self:assertValidProbs(inference_labels)
		elseif(initPoint == "pred") then
			local pred_labels = self.predictor_net:forward(inputs[2])
			inference_labels:copy(pred_labels):mul(1 - self.interpWeight):add(self.interpWeight*flatval)
		else
			inference_labels:fill(flatval)
		end

		--feed the features through once, and cache the results
		if(self.projector) then
			self.raw_features = self.projector:forward(inputs[2])
		else
			self.raw_features = inputs[2]	
		end
		self.features_net:forward(self.raw_features)
		local input_to_net = {inference_labels,self.features_net.output}

		if(self.computeTiming) then
			self.inferenceStartTime = sys.clock()
			self.elapsedFeatureTime = self.inferenceStartTime - self.startTime
		end

		return self:performInference(input_to_net,inference_labels,target_labels,loss_augmented,clampTrueLabels)
end


function Inference:performInference(input_to_net,inference_labels,target_labels,do_loss_augmented_inference,clampTrueLabels)
	if(do_loss_augmented_inference) then assert(target_labels ~=nil) end
    local parameters = inference_labels  
    local curErr = 1000000000000
    local prevErr = curErr
    local currOutput = nil
    local calls = 0
    local energyGradNorm = 0
    local gradNorm = 0
    if(self.verbose) then print('STARTING INFERENCE') end

    if(self.changingBatchSize and parameters:size(1) ~= self.fixedUnitGradient:size(1)) then 
    	self.fixedUnitGradient = torch.Tensor(parameters:size(1)):fill(1.0) 
    	if(self.cuda) then self.fixedUnitGradient = self.fixedUnitGradient:cuda() end
    end
    local peak = 0
    local optimConfig = Util:CopyTable(self.optConfig) --the optimizer may mutate the optConfig, so it's best to make a copy every time we start inference. this is a copy by value of the first layer
    local targ_for_structured_loss = nil
    if(do_loss_augmented_inference) then
	    targ_for_structured_loss = self.structured_training_loss.target_preprocess:forward(target_labels) 
	end

	local per_example_energy
	local per_example_energy_prev
	if(self.computePerExampleEnergy) then
		per_example_energy = self.fixedUnitGradient:clone()
		per_example_energy_prev = per_example_energy:clone()
	end

	self.noise = self.noise or inference_labels:clone():zero()
	self.noise:resizeAs(inference_labels)
    local function fEval(x)
    	local err = 0
    	calls = calls + 1
        if parameters ~= x then parameters:copy(x) end
        self.inference_net:zeroGradParameters() 

        if(not self.computePerExampleEnergy) then
	        err  = err + self.inference_net:forward(input_to_net):sum()
	    else
	    	per_example_energy_prev:copy(per_example_energy)
	    	per_example_energy:copy(self.inference_net:forward(input_to_net))
	    	err = err + per_example_energy:sum()
	    end
        assert(math.abs(self.fixedUnitGradient:mean() - 1.0 ) < 0.000001) --there were crazy bugs that set fixedUnitGradient to zero. keep this assertion for now
	    self.inference_net:updateGradInput(input_to_net, self.fixedUnitGradient) 
	    local cond = math.abs(self.fixedUnitGradient:mean() - 1.0 ) < 0.000001

       -- assert(cond,'mean = '..self.fixedUnitGradient:mean())
	    if(not cond) then
	    	self.fixedUnitGradient:fill(1.0)
	    end

      	local gradParametersTensor = self:getGradPrediction()


      	if(self.assert) then 
      		local ok = gradParametersTensor:eq(gradParametersTensor):all()
      		if(not ok) then 
      			print(self.inference_net)
      			print(currOutput)
      			print(input_to_net)

				print(input_to_net[1]:min()..' '..input_to_net[1]:max())
				print('----------------')
				
	  		end
      		assert(ok,"nans in gradient at inference iter "..calls) 
      	end --assert no nans in gradient 
      	--todo: do we really need to copy the gradient, since it gets overwritten here?
      	--note that this adds the loss gradient directly to inference_net.gradInput
      	if(do_loss_augmented_inference) then
      		--the training loss is defined on probabilities. 
      		local pred = self.structured_training_loss.prediction_preprocess:forward(parameters)
      		local label_loss = self.structured_training_loss.loss_criterion:forward(pred,targ_for_structured_loss)
      		err = err + (-self.structured_training_loss.loss_scale)*label_loss --subtract, rather than add, since our loss-augmented inference does minimization
      		if(self.computePerExampleEnergy) then
      			per_example_energy:add(-self.structured_training_loss.loss_scale,self.structured_training_loss.loss_evaluator(pred,targ_for_structured_loss))
      		end
      		local bg = self.structured_training_loss.loss_criterion:updateGradInput(pred,targ_for_structured_loss)
      		local pred_grad = self.structured_training_loss.prediction_preprocess:backward(parameters,bg)
      		gradParametersTensor:add(-self.structured_training_loss.loss_scale,pred_grad) -- add the negative gradient, since our loss-augmented inference does minimization
      	end

      	if(self.assert) then 
      		energyGradNorm = gradParametersTensor:norm()
	      	assert(gradParametersTensor:eq(gradParametersTensor):all(),'nans in inference gradient. current norm = '..gradParametersTensor:norm().." energy grad norm  = "..energyGradNorm) 
      	end 
    
      	err = err/parameters:size(1)
      	prevErr = curErr
      	curErr = err

      	if(self.verbose) then gradNorm = gradParametersTensor:max() end
      	if(self.renormGradients) then gradParametersTensor:renorm(1,gradParametersTensor:dim(),100) end

      	if(self.usePerturbation) then
      		torch.randn(self.noise,self.noise:size())
      		local scale = self.inferencePerturbation/math.sqrt(calls)
      		gradParametersTensor:add(scale,self.noise)
      	end
      	if(self.clipGradient) then 
      		local n = gradParametersTensor:norm()/gradParametersTensor:nElement()
      		if(n > self.inferenceGradientClip) then
      			--todo: this should really be: gradParametersTensor:div(n/self.inferenceGradientClip), but not changing for reproducibility's sake
      			gradParametersTensor:div(n)
      		end
      	end

        return err, gradParametersTensor
    end

     local j = 1
     local terminatedEarly = false
     local optimState = {}
     local prevParameters = parameters:clone()
     local numPredictionVars = parameters:nElement()/parameters:size(parameters:dim())
     local rowConvergenceObjective = torch.Tensor(parameters:size(1)):fill(self.numIters)
     local rowConvergenceIterate = rowConvergenceObjective:clone()
     local sign
     if(self.useL1) then
     	sign = parameters:clone()
     end

     local numExamples = parameters:size(1)
     local sizes_for_per_example_change = torch.LongStorage({numExamples,parameters:nElement()/numExamples})
     local info = {}
     local rowConvergence = self.fixedUnitGradient:clone():fill(0)

     local oneMinusTargetLabels
     if(clampTrueLabels and (not self.singleProbability)) then
	    oneMinusTargetLabels  = target_labels:clone():mul(-1):add(1)
	 end
	 
     local dd = parameters:dim()
     local function clampLabelsToTruth()
     	if(not self.singleProbability) then
	     	parameters:narrow(dd,2,1):cmax(target_labels:narrow(dd,2,1))
    	 	parameters:narrow(dd,1,1):cmax(oneMinusTargetLabels:narrow(dd,1,1))
     	else
     		parameters:cmax(target_labels)
     	end
     end
     if(clampTrueLabels) then clampLabelsToTruth() end
     while(true) do
         j = j + 1

        prevParameters:copy(parameters)
	    self.optimMethod(fEval, parameters,optimConfig,optimState)
	    if(clampTrueLabels) then clampLabelsToTruth() end

	    --this does an ista step on the predictions, shifting things to have fewer positive predictions
	    if(self.useL1) then
		    sign:copy(parameters):sign()
		    --only shrink the positive prediction
		    assert(optimState.currentLearningRate,'use an optimizer that sets this variable at every call')
		    local alpha = self.l1*optimState.currentLearningRate
		    if(not self.singleProbability) then
				parameters:narrow(3,2,1):abs():add(-alpha):cmax(0):cmul(sign:narrow(3,2,1))
				parameters:cdiv(parameters:sum(3):expandAs(parameters))
			else
				parameters:abs():add(-alpha):cmax(0):cmul(sign)
			end
		end


	    if(self.assert) then self:assertValidProbs(parameters) end 

		if(curErr ~= curErr) then
			print('NAN/INF in inference objective')
			print('curr inference objective = '..curErr)
			os.exit()
		end

	    if(self.verbose and  j % self.loggingFreq == 0) then
	    	local e,gradParameters = fEval(parameters)
		    print('inference objective: '..curErr)
		    print('peak = '..self:peakedness(parameters))
		    print('gradnorm = '..gradParameters:norm())
		end
		if(self.assert) then assert(parameters:eq(parameters):all(),'nans in inference beliefs after gradient step') end

		local numericallyConverged =false
		rowConvergence:fill(0)
		if(self.usePercentileForConvergence and j > 1) then
			per_example_energy_prev:add(-1,per_example_energy):cdiv(per_example_energy):abs()
			rowConvergence:copy(per_example_energy_prev:lt(self.objTol):squeeze())
			if(self.trackRowConvergence) then
				for i = 1,rowConvergenceObjective:size(1) do
					if((rowConvergenceObjective[i] == self.numIters) and (rowConvergence[i] == 1)) then
						rowConvergenceObjective[i] = j
					end
				end
			end
		else
			numericallyConverged = numericallyConverged or math.abs((curErr - prevErr)/curErr) < self.objTol
		end

	    
	    prevParameters:add(-1,parameters):abs()
		local numericallyConverged_iterate = false
		if(self.usePercentileForConvergence) then
			local per_example_change = prevParameters:view(sizes_for_per_example_change):max(2) 
			rowConvergence:add(per_example_change:lt(self.iterateTol):squeeze():typeAs(rowConvergence)):cmin(1)
			if(self.trackRowConvergence) then
				for i = 1,rowConvergenceIterate:size(1) do
					if((rowConvergenceIterate[i] == self.numIters) and (rowConvergence[i] == 1)) then
						rowConvergenceIterate[i] = j
					end
				end
			end
			numericallyConverged = rowConvergence:mean() > self.convergencePercentile
		else
		    local change = prevParameters:max()
			numericallyConverged = numericallyConverged or change < self.iterateTol
		end
		 
		local converged = (j == self.numIters) or (curErr <= prevErr and numericallyConverged)
		if(converged and j > self.minIters)then  
			if(self.trackRowConvergence) then
				info.iterateConvergenceIters = rowConvergenceIterate:cmin(j)
				info.objConvergenceIters = rowConvergenceObjective:cmin(j)
			end
			if(self.computeTiming) then
				local curTime  = sys.clock()
				info.totalTime = curTime - self.startTime
				info.inferenceTime = curTime - self.inferenceStartTime
				info.elapsedFeatureTime = self.elapsedFeatureTime
			end
			if(self.verbose) then  
				print('min necessary iters = '..rowConvergence:min())
				print('avg necessary iters = '..rowConvergence:mean())
				print('converged in '..j)
				peak = self:peakedness(parameters) 
				print('converged early after: '..j) 
				print('peakedness = '..peak)
				if(numericallyConverged_obj) then print('numerically converged obj') end
				if(numericallConverged_iterate) then print('numerically converged iterate: '..change.." vs. "..self.iterateTol) end
				if(numericallyConverged_boundary) then print('boundary convergence') end
			end
			terminatedEarly = true
			break 
		end
	end

	if(self.checkForPeak and self.assert and peak < self.peakednessThresh) then print("WARNING: returning non-peaked beliefs. Terminated after "..j.." iters. Peak = "..peak) end

	if(self.kinda_verbose and terminatedEarly) then 
		print('inference converged after '..j.." iters") 
	elseif(self.verbose) then
		print('terminating w/o convergence after '..j..' iters')
	end

	
	local prediction_to_return = parameters--:clone() --todo: this should be avoidable, since we allocate new parameters at the beginning of do_inference
	local inference_score_to_return =  self.inference_net:forward(input_to_net):clone() 
	if(do_loss_augmented_inference) then
		--note for historical reasons: the following commented line is buggy, but was used early on for some exploratory experiments
		--local loss_term = self.structured_training_loss.loss_evaluator(prediction_to_return,target_labels)
		local loss_term = self.structured_training_loss.loss_evaluator(self.structured_training_loss.prediction_preprocess:forward(prediction_to_return),targ_for_structured_loss)
		inference_score_to_return:add(-self.structured_training_loss.loss_scale,loss_term)
	end
	
	local necessaryIters = j
	info.necessaryIters = necessaryIters
	return inference_score_to_return, prediction_to_return, necessaryIters, info
end


function Inference:assertValidProbs(probs)
	local ds = probs:dim()
	local sums = probs:sum(ds):add(-1.0)
	assert(sums:abs():max() < 0.00001)
end
function Inference:getGradPrediction() return self.inference_net.gradInput[1] end 

--this calls the inference net, using the features that were most recently used in doInference()
--note: this expects proper, non-logged probabilities
function Inference:forwardOnLabels(labels)
	return self.inference_net:forward({labels,self.features_net.output}):clone()
end

function Inference:trainingLoss(pred,targ)
	return self.structured_training_loss.loss_evaluator(pred,targ)
end

function Inference:peakedness(probs) 
	local maxes = probs:max(probs:dim())
	return maxes:mean()
end

--this returns a number per minibatch example. TODO: right now it assumes b x T x L probs
function Inference:peakednessPerExample(probs) 
	local maxes = probs:max(probs:dim())
	return maxes
end

