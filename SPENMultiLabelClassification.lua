local SPENMultiLabelClassification,parent = torch.class('SPENMultiLabelClassification','SPENProblem')

function SPENMultiLabelClassification:__init(params)
	params.numClasses = params.labelDim
	params.labelDim = 2
	self.shuffle = params.shuffle == 1

	self.perLabelMode = params.factored == 1
	parent.__init(self,params)
	self.probabilityDim = 3
	self.binarylabels = true
	self.singleProbability = params.binaryInference == 1
	self.useMFPredictor = self.params.meanField == 1 
	self.params.numMFIters = self.params.meanFieldIters 
	self.uniform_init_value = 1/self.params.numClasses

	local predictionThreshes = {0,0.05,0.10,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.60,0.65,0.70,0.75}

	if(params.predictionThresh > 0) then predictionThreshes = {params.predictionThresh} end

	local analyzer
	local predictionType
	if(self.params.evaluateByRanking == 1) then
		predictionType = "rowWise"
	end
	self.evaluator = MultiLabelEvaluation(predictionThreshes,predictionType,analyzer,self.params.resultsFile) 

	parent.finalizeInit(self)

	self.classification_mapper = nn.Sequential()

	self.classification_mapper:add(nn.MulConstant(2.0))
	self.classification_mapper:add(nn.AddConstant(-1.0))
	if(self.params.useCuda) then self.classification_mapper:cuda() end

	self.iterate_transform = function() return nn.Sigmoid() end
end

function SPENMultiLabelClassification:multiPassMinibatcher(fileList,minibatch,useCuda,preprocess)
	return MinibatcherFromFileList(fileList,minibatch,useCuda,preprocess,self.shuffle)
end
function SPENMultiLabelClassification:onePassMinibatcher(fileList,minibatch,useCuda,preprocess)
	return OnePassMiniBatcherFromFileList(fileList,minibatch,useCuda,preprocess)
end

function SPENMultiLabelClassification:evaluateInference(testBatcher,inferencer,iterName)
	self.evaluator:evaluateInference(testBatcher, inferencer,self.params.testInferenceInitAtPred == 1,iterName)
end
function SPENMultiLabelClassification:evaluateClassifier(testBatcher,net,iterName)
	self.evaluator:evaluateClassifier(testBatcher,net,iterName)
end


function SPENMultiLabelClassification:preprocess(lab,feats,num)
	return lab,feats,num
end

function SPENMultiLabelClassification:getFeatureNetsImpl()
	local input_features_net
	if(not (self.params.linearFeatures == 1)) then
		local embeddingLayer = nn.Sequential()
		if(not (self.params.sparseFeatures == 1)) then
			embeddingLayer:add(nn.Linear(self.params.inputSize,self.params.embeddingDim))
	 	else
	 		error('this might not be supported still')
			embeddingLayer:add(nn.SparseLinearBatch(self.params.inputSize,self.params.embeddingDim,false))
	 	end

		self.embeddingLayer = embeddingLayer
		input_features_net = nn.Sequential()
		input_features_net:add(embeddingLayer)
		
		input_features_net:add(self:featuresNonlinearity())
	 	self:addDropout(input_features_net,false,{self.params.embeddingDim})

	 	input_features_net:add(nn.Linear(self.params.embeddingDim,self.params.featureDim))

	 	input_features_net:add(self:featuresNonlinearity()) --d: b x E
		
	else
		input_features_net = nn.Identity()
		self.embeddingLayer = input_features_net
		self.params.featureDim = self.params.inputSize
	end

	local input_classifier_layer
	if(self.params.linearFeatures == 1 and self.params.sparseFeatures == 1) then
		input_classifier_layer = nn.SparseLinearBatch(self.params.featureDim,self.params.numClasses,false) ---d: b  x L
	else
		print(self.params.featureDim,self.params.numClasses)
		input_classifier_layer = nn.Linear(self.params.featureDim,self.params.numClasses) ---d: b  x L
	end
	self.input_classifier_layer = input_classifier_layer
	local input_features_pretraining_net
	if(self.useMFPredictor) then
		input_features_pretraining_net = self:MFPredictor(input_features_net,self.params.featureDim)
	else
	 	input_features_pretraining_net= nn.Sequential()
		input_features_pretraining_net:add(input_features_net)
		input_features_pretraining_net:add(input_classifier_layer)

		if(self.params.lossType == "log") then
			input_features_pretraining_net:add(nn.Sigmoid())
		end
	end
	
	self.input_features_prediction_net = input_classifier_layer

	self.input_features_probability_prediction_net = nn.Sequential():add(input_classifier_layer):add(nn.Sigmoid())
	if(not self.singleProbability) then
		local expander = nn.Sequential()
		local branch1 = nn.Sequential()
		branch1:add(nn.MulConstant(-1))
		branch1:add(nn.AddConstant(1))
		branch1:add(nn.MyReshape(-1,-1,1))
		local branch2 = nn.MyReshape(-1,-1,1)
		expander:add(nn.ConcatTable():add(branch1):add(branch2)):add(nn.JoinTable(3,3))
		self.input_features_probability_prediction_net:add(expander)
	end

	self.input_features_net = input_features_net 
	self.input_net = input_net 
	self.input_features_pretraining_net = input_features_pretraining_net 
	self.modulesToOptimize = modulesToOptimize 
	self.label_net_for_pretraining = nn.Sequential()



	if(not self.singleProbability) then
		self.prediction_preprocess = nn.Select(3,2)
		self.target_preprocess = nn.Select(3,2)
	else
		self.prediction_preprocess = nn.Identity()
		self.target_preprocess = nn.Identity() 
	end
end

function SPENMultiLabelClassification:copyParameters(target,source)
	local tp = target:parameters()
	local sp = source:parameters()
	for i,t in ipairs(tp) do
		t:copy(sp[i])
	end
end
function SPENMultiLabelClassification:loadPretrainNet(pretrainNet)
	if(not self.useMFPredictor) then
		parent.loadPretrainNet(self,pretrainNet)
	else
		print('loading pretrained unary potentials from '..pretrainNet)
		local loaded = torch.load(pretrainNet):double()
		  local features_net = loaded.modules[1]
		  local unary_net = loaded.modules[2]
		 self:copyParameters(self.unary_potentials_net,unary_net)
		 self:copyParameters(self.input_features_net,features_net)
	end
end



function SPENMultiLabelClassification:getLabelEnergyNetImpl()
	local label_energy_net = nn.Sequential()
	if(not self.singleProbability) then
		label_energy_net:add(nn.Select(3,2))
	end

	local shiftLabels = true --this is unnecessary in terms of representation capacity of the network, but it might be easier to learn things if we do this. plus, it's more symmetric in terms of strongly preferring a label setting vs. strongly disliking one. Otherwise, 0 might not be special enough of a value (for a label being off)
	if(shiftLabels) then
		local shifter = nn.Sequential():add(nn.MulConstant(2,true)):add(nn.AddConstant(-1,true))
		label_energy_net:add(shifter)
	end

	local eT = self.params.labelEnergyType
	if(not self.useConditionalLabelEnergy) then
		if(eT == "allPairs") then
			local nC = self.params.numClasses
			label_energy_net:add(nn.SelfOuterProd(nC,nC))
			label_energy_net:add(nn.Linear(nC*nC,1))
		elseif(eT == "lowRankAllPairs") then
			local nC = self.params.energyDim
			label_energy_net:add(nn.Linear(self.params.numClasses,self.params.energyDim)) 
			label_energy_net:add(nn.SelfOuterProd(nC,nC))
			label_energy_net:add(nn.Linear(nC*nC,1))
		elseif(eT == "deep") then
			self:addDropout(label_energy_net,false,{self.params.numClasses})
			label_energy_net:add(nn.Linear(self.params.numClasses,self.params.energyDim)) 
			label_energy_net:add(self:energyNonlinearity())
			self:addDropout(label_energy_net,true,{self.params.energyDim})
			label_energy_net:add(nn.Linear(self.params.energyDim,1)) 
		elseif(eT == "deeper") then
			self:addDropout(label_energy_net,false,{self.params.numClasses})
			label_energy_net:add(nn.Linear(self.params.numClasses,self.params.energyDim)) 
			label_energy_net:add(self:energyNonlinearity())
			self:addDropout(label_energy_net,true,{self.params.energyDim})
			label_energy_net:add(nn.Linear(self.params.energyDim,self.params.energyDim)) 
			label_energy_net:add(self:energyNonlinearity())
			self:addDropout(label_energy_net,true,{self.params.energyDim})
			label_energy_net:add(nn.Linear(self.params.energyDim,1)) 
		elseif(eT == "measurements") then
			local measurements = nn.Linear(self.params.numClasses,self.params.energyDim)
			label_energy_net:add(measurements) 
			label_energy_net:add(self:energyNonlinearity())
			local cmul = nn.CMul(self.params.energyDim)
			label_energy_net:add(cmul) 

			label_energy_net:add(nn.Sum(2))
			label_energy_net:add(nn.Reshape(1,true))
		else
			assert(false,"invalid energy type")
		end
		if(self.params.initUnaryWeight ~= 1.0) then label_energy_net:add(nn.MulConstant(1/self.params.initUnaryWeight,true)) end

		self.label_energy_net = label_energy_net

	end


	if(self.useConditionalLabelEnergy) then
		--the input is a table where the first arg is the labels and the second are features
		local conditional_label_energy_net  = nn.Sequential()
		local branch1 = nn.Sequential() --this is for the labels
		if(not self.singleProbability) then
			branch1:add(nn.Select(3,2)) 
		end

		local branch2 = nn.Sequential()

		local eT = self.params.labelEnergyType
		if(eT == "lowRankAllPairs") then
			branch1:add(nn.Linear(self.params.numClasses,self.params.energyDim))
			branch1:add(self:energyNonlinearity())
			branch2:add(nn.Linear(self.params.featureDim,self.params.energyDim))
			branch2:add(self:energyNonlinearity())
			conditional_label_energy_net:add(nn.ParallelTable():add(branch1):add(branch2))

			conditional_label_energy_net:add(nn.JoinTable(2,2))
			conditional_label_energy_net:add(nn.SelfOuterProd(2*self.params.energyDim))
			conditional_label_energy_net:add(nn.Linear(4*self.params.energyDim*self.params.energyDim,1))
		elseif(eT == "concat" or eT == "deep") then
			self:addDropout(branch1,false,{self.params.numClasses})
 			branch1:add(nn.Linear(self.params.numClasses,self.params.energyDim))
			branch1:add(self:energyNonlinearity())

			self:addDropout(branch2,false,{self.params.featureDim})
			branch2:add(nn.Linear(self.params.featureDim,self.params.energyDim))
			branch2:add(self:energyNonlinearity())

			conditional_label_energy_net:add(nn.ParallelTable():add(branch1):add(branch2))

			conditional_label_energy_net:add(nn.JoinTable(2,2))	
			self:addDropout(conditional_label_energy_net,true,{2*self.params.energyDim})		
			conditional_label_energy_net:add(nn.Linear(2*self.params.energyDim,self.params.energyDim))
			conditional_label_energy_net:add(self:energyNonlinearity())
			self:addDropout(conditional_label_energy_net,true,{self.params.energyDim})		

			conditional_label_energy_net:add(nn.Linear(self.params.energyDim,1))
		elseif(eT == "deeper" or eT == "deepest") then
			self:addDropout(branch1,false,{self.params.numClasses})		
 			branch1:add(nn.Linear(self.params.numClasses,2*self.params.energyDim))
			branch1:add(self:energyNonlinearity())

			self:addDropout(branch2,false,{self.params.featureDim})		
			branch2:add(nn.Linear(self.params.featureDim,2*self.params.energyDim))
			branch2:add(self:energyNonlinearity())

			conditional_label_energy_net:add(nn.ParallelTable():add(branch1):add(branch2))

			conditional_label_energy_net:add(nn.JoinTable(2,2))	
			self:addDropout(conditional_label_energy_net,true,{4*self.params.energyDim})							
			conditional_label_energy_net:add(nn.Linear(4*self.params.energyDim,2*self.params.energyDim))
			conditional_label_energy_net:add(self:energyNonlinearity())
			self:addDropout(conditional_label_energy_net,true,{2*self.params.energyDim})					
			conditional_label_energy_net:add(nn.Linear(2*self.params.energyDim,self.params.energyDim))
			conditional_label_energy_net:add(self:energyNonlinearity())
			if(eT == "deepest") then
				self:addDropout(conditional_label_energy_net,true,{self.params.energyDim})		
				conditional_label_energy_net:add(nn.Linear(self.params.energyDim,self.params.energyDim))
			end
			self:addDropout(conditional_label_energy_net,true,{self.params.energyDim})		
			conditional_label_energy_net:add(nn.Linear(self.params.energyDim,1))
		elseif(eT == "linear") then
 			branch1:add(nn.Identity())
			branch2:add(nn.Linear(self.params.featureDim,self.params.numClasses))

			conditional_label_energy_net:add(nn.ParallelTable():add(branch1):add(branch2))
			conditional_label_energy_net:add(nn.CMulTable(2,2))	
			conditional_label_energy_net:add(nn.Sum(2))		
		else
			assert(false,'invalid labelEnergyType')
		end
		if(self.params.initUnaryWeight ~= 1.0) then conditional_label_energy_net:add(nn.MulConstant(1/self.params.initUnaryWeight,true)) end
		self.conditional_label_energy_net = conditional_label_energy_net

	end

end

function SPENMultiLabelClassification:MFPredictor(features_net,numFeatures)
	local autograd = require 'autograd'
	autograd.protected(true)
	autograd.optimize(true)
	require 'Diag'	

	local numLabels = self.params.numClasses
	local diag = autograd.functionalize(nn.Diag())
	local sigmoid = autograd.functionalize(nn.Sigmoid())

	-----------------------------------------------------------------------------
	-----------Networks for Getting Unary and Pairwise Potentials----------------------
	local pairwisePotentials = nn.Sequential()
	local dataIndependent = self.params.dataIndependentMeanField == 1
	if(dataIndependent) then
		pairwisePotentials:add(nn.MulConstant(0)) --obviously this isn't the best thing to do computationally
	end
	local pairwisePotentialsFunction = nn.Linear(numFeatures,numLabels*numLabels)

	--initialize them really really small
	for _,p in ipairs(pairwisePotentialsFunction:parameters()) do
		p:zero()
	end

	pairwisePotentials:add(pairwisePotentialsFunction):add(nn.View(numLabels,numLabels))
	
	self.pairwisePotentials = pairwisePotentials
	local unaryPotentials = nn.Linear(numFeatures,numLabels)
	self.unary_potentials_net = unaryPotentials
	
	local par = nn.ConcatTable():add(pairwisePotentials):add(unaryPotentials)
	local potentialsNet = nn.Sequential():add(features_net):add(par)
	-----------------------------------------------------------------------------

	---------------------------------Mean-Field Layer----------------------------
	local mm = autograd.functionalize(nn.MM())

	local labelReshape =autograd.functionalize(nn.Reshape(numLabels,1))

	local MF = function(tab)
		local labelBeliefs,potentials = unpack(tab)
		local A,U = unpack(potentials)
		local D = diag(A)
		local labelBeliefs2 = labelReshape(labelBeliefs)
		local S = mm({A,labelBeliefs2}) - torch.cmul(D,labelBeliefs2)
		local E = D + S + U 
		return sigmoid(E)
	end

	function MFLayer()
		local cat = nn.ConcatTable():add(autograd.nn.AutoModule('MF')(MF)):add(nn.SelectTable(2))
		local MFLayerNet = nn.Sequential():add(cat)
		return MFLayerNet
	end
	-----------------------------------------------------------------------------

	---------------------------------Mean-Field Network----------------------------
	local cat2 = nn.ParallelTable():add(nn.Identity()):add(potentialsNet)
	local net = nn.Sequential():add(cat2)
	for i = 1,self.params.numMFIters do
		net:add(MFLayer())
	end
	net:add(nn.SelectTable(1))
	-----------------------------------------------------------------------------

	local net2 = nn.Sequential()
	local MFNet, parent2 =  torch.class('nn.MFNet', 'nn.Sequential')
	function MFNet:__init(dual_net,numLabels)
   		parent2.__init(self)
   		self:add(dual_net)
   		self.dual_net = dual_net
   		self.numLabels = numLabels

	end
	function MFNet:forward(input)
		self.initBeliefs = self.initBeliefs or input.new()
		self.initBeliefs:resize(input:size(1),self.numLabels)
		self.initBeliefs:fill(0.5)
		return self.dual_net:forward({self.initBeliefs,input})
	end
	function MFNet:backward(input,gradOutput)
		self.initBeliefs:fill(0.5)
		return self.dual_net:backward({self.initBeliefs,input},gradOutput)
	end
	function MFNet:__tostring__()
		return self.dual_net:__tostring__()
	end

	return nn.MFNet(net,numLabels)
end


