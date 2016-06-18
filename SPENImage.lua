local SPENImage,parent = torch.class('SPENImage','SPENProblem')

function SPENImage:__init(params)
	params.numClasses = params.labelDim
	params.labelDim = 2
	self.shuffle = params.shuffle == 1
	self.nn = nn
	if(params.cudnn == 1) then
		print('using cudnn')
		require 'cudnn'
		self.nn = cudnn
	end
	self.perLabelMode = params.factored == 1
	self.singleProbability = 1
	parent.__init(self,params)
	self.probabilityDim = 4
	self.binarylabels = true
	local predictionThreshes = {0,0.05,0.10,0.15,0.2,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.60,0.65,0.70,0.75}

	if(params.predictionThresh > 0) then predictionThreshes = {params.predictionThresh} end
	--todo: what's the API for this?
	self.evaluator = ImageEvaluation(predictionThreshes,predictionType,analyzer,self.params.resultsFile) 
	self.classification_mapper = nn.Identity()--nn.Select(3,1)
	if(self.params.useCuda) then self.classification_mapper:cuda() end
	parent.finalizeInit(self)
	self.structured_training_loss.loss_criterion.sizeAverage = true

end

function SPENImage:multiPassMinibatcher(fileList,minibatch,useCuda,preprocess)
	return MinibatcherFromFileList(fileList,minibatch,useCuda,preprocess,self.shuffle)
end

function SPENImage:onePassMinibatcher(fileList,minibatch,useCuda,preprocess)
	return OnePassMiniBatcherFromFileList(fileList,minibatch,useCuda,preprocess)
end

function SPENImage:evaluateClassifier(testBatcher,net,iterName)
	self.evaluator:evaluateClassifier(testBatcher,net,iterName)
end


function SPENImage:classificationPreprocess(lab,feats,num)
	local lab2 = self.classification_mapper:forward(lab):clone()
	return lab2,feats,num
end

function SPENImage:preprocess(lab,feats,num)
	if(self.params.averageLoss == 1) then
		error('not implemented yet')
	end
	return lab,feats,num
end

function SPENImage:getFeatureNetsImpl()
	self.outputFrameSize = self.params.featureDim
	local featureDim = self.outputFrameSize
	local input_features_net = nn.Sequential()
	local convWidth = 5 --todo: use this below
	local padSize = (convWidth - 1)/2
	local deep = self.params.deepFeatures == 1

	local stride = 1
	if(deep) then
		input_features_net:add(self.nn.SpatialConvolution(3,featureDim,5,5,1,1,2,2))
		input_features_net:add(self:featuresNonlinearity())
		self:addDropout(input_features_net)

		input_features_net:add(self.nn.SpatialConvolution(featureDim,featureDim,5,5,stride,stride,2,2))
		input_features_net:add(self:featuresNonlinearity())
		self:addDropout(input_features_net)

		if(stride > 1) then
			input_features_net:add(nn.SpatialFullConvolution(featureDim,featureDim,5,5,stride,stride,padSize,padSize,3,3))
			self:addDropout(input_features_net)
		end

		input_features_net:add(nn.SpatialConvolution(featureDim,self.params.finalFeatureDim,5,5,1,1,2,2))
		input_features_net:add(self:featuresNonlinearity())
	else 
		input_features_net:add(self.nn.SpatialConvolution(3,self.params.finalFeatureDim,5,5,1,1,2,2))
		input_features_net:add(self:featuresNonlinearity())
	end


	local input_features_base_net = nn.Sequential()
	local input_classifier_layer = self.nn.SpatialConvolution(self.params.finalFeatureDim,1,1,1,1,1,0)
	input_features_base_net:add(input_classifier_layer)

	local input_features_pretraining_net  = nn.Sequential():add(input_features_net):add(input_features_base_net):add(nn.Select(2,1))
	if(self.useBCEClassificationCriterion) then
		input_features_pretraining_net:add(nn.Sigmoid())
	end

	self.input_features_prediction_net = nn.Sequential():add(input_features_base_net):add(nn.Select(2,1))
	--[[
	local expander = nn.Sequential()
	local branch1 = nn.Sequential()
	branch1:add(nn.MulConstant(-1))
	branch1:add(nn.AddConstant(1))
	local branch2 = nn.Identity()
	expander:add(nn.ConcatTable():add(branch1):add(branch2)):add(nn.JoinTable(3,3)) --todo: change this to 4-4 when we minibatch
	--]]
	self.input_features_probability_prediction_net = nn.Sequential():add(input_features_base_net):add(nn.Sigmoid())

	local input_net = nn.Sequential()
	self.input_features_net = input_features_net 
	self.input_net = input_net 
	self.input_features_pretraining_net = input_features_pretraining_net 
	self.modulesToOptimize = modulesToOptimize 
	self.label_net_for_pretraining = nn.Sequential()

	self.prediction_preprocess = nn.Identity()
	self.target_preprocess = nn.Identity() 
	self.iterate_transform = function() return nn.Sigmoid() end

	--self.prediction_preprocess = nn.Select(3,2) 
	--self.target_preprocess = nn.Select(3,2)
end

function SPENImage:getEnergyNetImpl()
	self.energy_net = nn.Sequential()
	self.energy_net_per_label = nn.Sequential()
end


function SPENImage:getLabelEnergyNetImpl()
	local energyWidth = 5
	local padSize = (energyWidth - 1)/2
	local label_energy_net = nn.Sequential()
	label_energy_net:add(nn.Replicate(1,2))
	label_energy_net:add(nn.Copy(nil,nil,true))
	label_energy_net:add(self.nn.SpatialConvolution(1, self.params.energyDim, energyWidth,energyWidth,1,1))
	label_energy_net:add(self:energyNonlinearity())
	label_energy_net:add(self.nn.SpatialConvolution(self.params.energyDim,1,1,1))

	--todo: this is inefficient
	label_energy_net:add(nn.Flatten()):add(nn.Sum(2))
	
	self.label_energy_net = label_energy_net

	local conditional_label_energy_net = nn.Sequential()
	local branch1 = nn.Sequential() --this is for the labels

	branch1:add(self.nn.SpatialConvolution(1, self.params.energyDim, energyWidth,energyWidth,1,1,padSize,padSize))

	local branch2 = nn.Identity() --this is for the features


	conditional_label_energy_net:add(nn.ParallelTable():add(branch1):add(branch2))
	conditional_label_energy_net:add(nn.JoinTable(2,4))			
	conditional_label_energy_net:add(self:featuresNonlinearity())
	conditional_label_energy_net:add(nn.SpatialConvolution(self.params.energyDim + self.params.finalFeatureDim,self.params.energyDim,1,1,1,1))
	conditional_label_energy_net:add(self:featuresNonlinearity())
	conditional_label_energy_net:add(nn.SpatialConvolution(self.params.energyDim,1,1,1,1,1))
	conditional_label_energy_net:add(nn.Flatten()):add(nn.Sum(2))




	self.conditional_label_energy_net = conditional_label_energy_net
end

