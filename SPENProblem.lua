local SPENProblem = torch.class('SPENProblem')

function SPENProblem:__init(params)
	self.params = params
	self.useDropout = params.dropout > 0
	self.useProjection = params.project == 1
	
	------Members expected to be accessed from outside
	self.initFullModel = nil --this is whether the full model was initialized from file
	self.useLabelEnergy = self.params.labelEnergy == 1
	self.useConditionalLabelEnergy = self.params.conditionalLabelEnergy == 1
	self.useDirectEnergy = self.params.directEnergy == 1
	self.binarylabels = false
	self.tableFeatures = false
	self.numOptimizationVariables = 1
	----
	---members to be accessed only privately
	self.fixedUnitGradientSize = torch.LongStorage(2)
	self.fixedUnitGradientSize[1] = 0 --this will be filled in as the minibatch size
	self.fixedUnitGradientSize[2] = 1 --this may be overwritten in subclasses
	--
	self.dropouts = {}
	self.prediction_selector = function(x) return x end
	self.learned_features_net = nil --this is used when we want to clamp lower-level features, but still learn some
end

---------------------------Abstract Methods----------------------------
--TODO: document all of these
--TODO: just change these to evaluate()
function SPENProblem:evaluateClassifier(testBatcher,net,iterName)
		self:abstractError()
end

function SPENProblem:evaluateInference(testBatcher,inferencer,iterName)
	self:abstractError()
end

function SPENProblem:preprocess(lab,feats,num)
	self:abstractError()
end

function SPENProblem:multiPassMinibatcher(fileList,minibatch,useCuda,preprocess)
	self:abstractError()
end
function SPENProblem:onePassMinibatcher(fileList,minibatch,useCuda,preprocess)
	self:abstractError()
end

--this isn't an abstract method, but you may want to add more things to it.
function SPENProblem:reset()
	for k,d in pairs(self.dropouts) do
		d:reset()
	end
end
------------------------------------------------------------------------

function SPENProblem:loadPretrainNet(pretrainNet)
		print('loading input features pretrained from '..pretrainNet)
		local loaded= torch.load(pretrainNet)
		self.input_features_pretraining_net:getParameters():copy(loaded:getParameters())  --is it ok to call getParameters like this?
end

--this is expected to be called at the end of child:__init()
function SPENProblem:finalizeInit()	
	self:getFeatureNets()
	if(self.useLabelEnergy or self.useConditionalLabelEnergy) then self:getLabelEnergyNetImpl() end
	self:getProcessingNets()

	local loadPretrain = self.params.initFeaturesNet and self.params.initFeaturesNet ~= ""
	local loadFullModel = self.params.initFullNet  and self.params.initFullNet ~= ""

	assert(not (loadPretrain and loadFullModel))

	if(loadPretrain) then
		self:loadPretrainNet(self.params.initFeaturesNet)
	end

	if(self.params.lossType) then
		self.structured_training_loss = self:initStructuredTrainingLoss()
	end

	local initAtLocalPrediction = self.params.initAtLocalPrediction == 1

	--this takes	{raw_features,features}. The idea is that it needs the raw_features to grab the right size of things
	self.initialization_net = nn.Sequential()

	if(initAtLocalPrediction) then
	 	self.initialization_net:add(nn.SelectTable(2))
	 	local predictor_net = self.input_features_probability_prediction_net --todo: this is a bad name
	 	local local_predictor = (self.params.clonePredictor == 1) and predictor_net:clone() or predictor_net
	 	self.initialization_net:add(local_predictor)
	else
		self.initialization_net:add(nn.SelectTable(1))
		self.initialization_net:add(nn.MulConstant(0)):add(nn.AddConstant(self.uniform_init_value)) --todo: this is doing extra work. should just make a ConstantFromSize layer or something...
	end

	local function toCuda(x) if(x) then x:cuda() end end
	if(self.params.useCuda) then
		toCuda(self.inference_net)
		if(self.params.lossType) then toCuda(self.structured_training_loss.loss_criterion) end
		toCuda(self.training_net)
		toCuda(self.structured_training_loss.target_preprocess)
		toCuda(self.structured_training_loss.prediction_preprocess)
		toCuda(self.label_net_for_pretraining)
		toCuda(self.input_features_pretraining_net)
		toCuda(self.input_features_prediction_net)
		toCuda(self.input_features_probability_prediction_net)
		toCuda(self.label_energy_net)
		toCuda(self.conditional_label_energy_net)
	end
end


function SPENProblem:initStructuredTrainingLoss()
	local lossType = self.params.lossType

	if(lossType == "ranking") then
		error('make sure this works in binary inference mode')
		loss_criterion = nn.MultiLabelRankCriterion(false,self.params.positiveWeight)
	elseif(lossType == "spos") then
		loss_criterion = nn.SPosCriterion(self.params.positiveWeight)
	elseif(lossType == "mse") then
		if(self.params.positiveWeight ~= 1.0) then
			loss_criterion = nn.MSECriterion()
		else
			if(not self.binarylabels) then
				loss_criterion = nn.ScaledMSECriterion(self.params.positiveWeight,2) --the 2 here is bc the positive class is assumed to be 2 
			else
				loss_criterion = nn.ScaledMSECriterion(self.params.positiveWeight) 
			end
		end
	elseif(lossType == "log") then
		self.useBCEClassificationCriterion = true
		if(self.params.positiveWeight ~= 1.0) then
			loss_criterion = nn.BCECriterion()
		else
			error('make sure this is compatible with the new way that we define positive weight')
			loss_criterion = nn.ScaledBCECriterion(self.params.positiveWeight) 
		end
	else
		error('invalid loss')
	end

	if(self.params.averageLoss == 1) then
		loss_criterion = nn.RepeatedCriterion(loss_criterion)
	end

	local structured_training_loss = {
		prediction_preprocess = self.prediction_preprocess, --todo: if this and target_preprocess aren't nn.Identity(), then they should be pulled into a TableCriterion combined with loss_criterion. 
		target_preprocess = self.target_preprocess,
		loss_criterion = loss_criterion, 
	}

	return structured_training_loss
end




function SPENProblem:classificationMinibatcher(fileList,onePass)
	assert(fileList)
	if(not onePass) then 
		return self:multiPassMinibatcher(fileList,self.params.minibatch,self.params.useCuda,function (x,y,z) return self:classificationPreprocess(x,y,z) end) --todo: fix this syntax
	else
		return self:onePassMinibatcher(fileList,self.params.testMinibatchSize,self.params.useCuda,function (x,y,z) return self:classificationPreprocess(x,y,z) end)
	end
end

function SPENProblem:minibatcher(fileList,onePass)
	assert(fileList)
	if(not onePass) then 
		return self:multiPassMinibatcher(fileList,self.params.minibatch,self.params.useCuda,function (x,y,z) return self:preprocess(x,y,z) end)
	else
		return self:onePassMinibatcher(fileList,self.params.testMinibatchSize,self.params.useCuda,function (x,y,z) return self:preprocess(x,y,z) end)
	end
end

function SPENProblem:getFeatureNets()
	self:getFeatureNetsImpl()

	return {self.input_features_net,self.input_net,self.input_features_pretraining_net,self.modulesToOptimize}
end

--this function could be cleaned up quite a bit, particularly by using nngraph
function SPENProblem:getProcessingNets()
	--this is for processing Q distributions that are on the simplex. 
	--It doesn't include feature computation, since this is assumed to be precomputed somewhere else.
	local inference_net = nn.Sequential()

	local fixed_features_net = nn.Sequential() --misnomer. these are the low-level features
	local input_features_net_to_use = self.useProjection and nn.Identity() or self.input_features_net 

	fixed_features_net:add(input_features_net_to_use)

	local function cnt(x) return x and 1 or 0 end
	local energyCount = cnt(self.useLabelEnergy) + cnt(self.useConditionalLabelEnergy) + cnt(self.useDirectEnergy)

	---This is for evaluating the energy given a specific value for the labels, but the raw features
	local training_net = nn.Sequential()
	local par = nn.ParallelTable()
	par:add(nn.Identity()) --this is for the labels
	par:add(fixed_features_net)
	training_net:add(par)

	self.modules_without_feature_layer = {}

	if(self.params.learnUnaryInFirstPass == 1) then
		table.insert(self.modules_without_feature_layer,self.input_classifier_layer)
	end

	local full_energy_nets = nn.ConcatTable()
	

	if(self.useDirectEnergy) then
		local directEnergyNet = nn.Sequential()	

		local par = nn.ParallelTable()
		local label_branch = nn.Sequential()

		local shifter = nn.Sequential():add(nn.MulConstant(2)):add(nn.AddConstant(-1))
		label_branch:add(shifter)


		par:add(label_branch)
		par:add(self.input_features_prediction_net)

		directEnergyNet:add(par)
		directEnergyNet:add(nn.CMulTable())

		if(not self.binarylabels) then		
			directEnergyNet:add(nn.Sum(2))
		end

		assert(self.singleProbability,'the next line needs to be generalized if not in single probability mode')
		directEnergyNet:add(nn.Reshape(self.params.numClasses))
		directEnergyNet:add(nn.Sum(2))

		directEnergyNet:add(nn.MulConstant(-1.0))
		if(self.params.scaleDirectEnergy == 1) then
			local scale = nn.Mul()
			directEnergyNet:add(scale)
			table.insert(self.modules_without_feature_layer,scale)
		end
		full_energy_nets:add(directEnergyNet)
	end

	if(self.useLabelEnergy) then
		local lab = nn.Sequential()
		lab:add(nn.SelectTable(1))
		lab:add(self.label_energy_net)
		full_energy_nets:add(lab)
		table.insert(self.modules_without_feature_layer,self.label_energy_net)
	end

	if(self.useConditionalLabelEnergy) then
		full_energy_nets:add(self.conditional_label_energy_net)
		table.insert(self.modules_without_feature_layer,self.conditional_label_energy_net)
	end

	local full_energy_net = nn.Sequential()
	full_energy_net:add(full_energy_nets)

	full_energy_net:add(nn.CAddTable())

	for i,net in ipairs({inference_net,training_net}) do
		net:add(full_energy_net)
	end


	self.training_net = training_net
	self.inference_net = inference_net
	self.fixed_features_net = fixed_features_net
end

function SPENProblem:addBatchNorm(net,size)
	if(self.params.batchnorm) then
		net:add(nn.SpatialBatchNormalization(size))
	end
end

function SPENProblem:addDropout(n,inplace,sizes)
	local ip
	if(inplace == nil) then
		ip = true
	else
		ip = inplace
	end
    if(self.useDropout) then --todo: change back
    	local drop = nn.EpochDropout(self.params.dropout,false,sizes,ip)
    	table.insert(self.dropouts,drop)
        n:add(drop)
    end
end



function SPENProblem:nonlinearity(name)
	if(name == "SoftPlus")then
		return nn.SoftPlus(4)
	elseif(name == "HardTanh")then
		return nn.HardTanh()
	elseif(name == "ReLU") then
		return nn.ReLU()
	elseif(name == "Sigmoid") then
		return nn.Sigmoid()
	elseif(name == "Identity") then
		return nn.Identity()
	elseif(name == "Square") then
		return nn.Square()
	else
		assert(false,"invalid nonlinearity: "..self.params.featuresNonlinearity)
	end
end

function SPENProblem:featuresNonlinearity()
	return self:nonlinearity(self.params.featuresNonlinearity)
	
end

function SPENProblem:energyNonlinearity()
	return self:nonlinearity(self.params.energyNonlinearity)
end

function SPENProblem:abstractError()
	assert(false,'to be implemented by classes that extend this')
end

