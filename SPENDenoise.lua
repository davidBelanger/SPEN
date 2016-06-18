local SPENDenoise,parent = torch.class('SPENDenoise','SPENProblem')


function SPENDenoise:__init(params)
	params.numClasses = params.labelDim
	params.labelDim = 2
	self.shuffle = params.shuffle == 1

	self.singleProbability = 1

	self.useCudaForData = true
	parent.__init(self,params)

	self.params.likKernelSize = params.priorKernelSize

	self.params.numInputChannels = 1 --hard coding this for now because some things need to be adapted slightly to support color inputs
	self.iterateRange = {0,1} --this means that the iterates are constrained to [0,1]. The alternative, for example, could be [0,255]
	self.uniform_init_value = 0.5*(self.iterateRange[2] + self.iterateRange[1])

	self.params.imageSize = 100 --We don't use this unless we use FFT features. In general, this part of the code is under-developed. 

	self.evaluator = PSNREvaluation(self.params.resultsFile,self.params.resultsFile ~= "")
	parent.finalizeInit(self)

	self.structured_training_loss.loss_criterion.sizeAverage = true

	if(self.params.writeImageExamples > 0) then
		self.imageAnalyzer = ImageAnalysis(self.params.imageExampleDir,5)
	end

	if(self.params.initEnergy ~= "") then
		print('loading from '..self.params.initEnergy)
		self.inference_net:getParameters():copy(torch.load(self.params.initEnergy):getParameters())
	end
end

function SPENDenoise:imageAnalysis(testBatcher,net,i)
	return self.imageAnalyzer:makeExamples(testBatcher,net,i)
end

function SPENDenoise:multiPassMinibatcher(fileList,minibatch,useCuda,preprocess)
	return SingleBatcher(fileList,minibatch,false,true,preprocess)
 end


function SPENDenoise:onePassMinibatcher(fileList,minibatch,useCuda,preprocess)
	return SingleBatcher(fileList,minibatch,true,true,preprocess)
end

function SPENDenoise:evaluateClassifier(testBatcher,net,iterName)
	self.evaluator:evaluateClassifier(testBatcher,net,iterName)
end


function SPENDenoise:preprocess(lab,feats,num)
	return lab,feats,num
end

function SPENDenoise:getFeatureNetsImpl()

	self.input_features_net = nn.Identity()

	self.prediction_preprocess = nn.Identity()
	self.target_preprocess = nn.Identity() 
end

function SPENDenoise:getEnergyNetImpl()
	
end

function SPENDenoise:getProcessingNets()
	self.energy_net = nn.Sequential()
	self.energy_net_per_label = nn.Sequential()
 	
 	if(not (self.params.gaussianBlurInit == 1)) then
 		self.input_features_probability_prediction_net = nn.Identity()  
	else 
		assert(self.params.numInputChannels == 1, 'right now this is only figured for grayscale input')
		local blur_net = nnlib.SpatialConvolution(1,1,5,5,1,1,2,2)
		local kernel = image.gaussian({normalize = true,size=5})
		blur_net.weight:copy(kernel:view(1,1,5,5))
		self.blur_net = blur_net
		self.input_features_probability_prediction_net = nn.TruncatedBackprop():add(blur_net)
	end
	self.fixed_features_net = nn.Identity()
end

function SPENDenoise:squared_loss(x,y) 
	local diff = nn.CSubTable()({x,y})
	local sq = nn.Square()(diff)
	local rs = nn.View(-1):setNumInputDims(2)(sq) 
	return nn.Sum(2)(rs)
end

function SPENDenoise:scaled_squared_loss(x,y) 
	local diff = nn.CSubTable()({x,y})
	local sq = nn.Square()(diff)
	return self:coordinate_scaled_sum(sq)
end

function SPENDenoise:coordinate_scaled_sum(input)
	local net = nn.Sequential()
	net:add(nn.CMul(self.params.imageSize,self.params.imageSize))
	net:add(nn.View(-1):setNumInputDims(2))
	net:add(nn.Sum(2))
	return net(input)
end

function SPENDenoise:basic_spatial_filter_response(im_predict)
	local filter_response_net = nn.Sequential()
	local conv = nnlib.SpatialConvolution(self.params.numInputChannels,self.params.energyDim,self.params.likKernelSize,self.params.likKernelSize,1,1)
	self:initializeConvWeights(conv)
	filter_response_net:add(conv)

	--see, for example, (23) in Domke, 2012
	filter_response_net:add(self:rho_squared())

	local reshaped_filter_responses = nn.View(-1):setNumInputDims(3)(filter_response_net(im_predict))
	return nn.Sum(2)(reshaped_filter_responses)
end

--note that initialization of the first layer of filters is very important. Domke 2012, and the Fields of Experts papers he extends,
--build up the filter bank incrementally. For convenience, we don't do this. However, initializing with appropriately diverse filters is challenging
function SPENDenoise:initializeConvWeights(conv)
	local init_template = torch.Tensor(5,5):fill(0) 
	for i = 2,4 do
		for j = 2,4 do
			init_template[i][j] = -1
		end
	end
	init_template[3][3] = 8 
	for i = 1,conv.weight:size(1) do
		local example = init_template:clone():add(0.25,torch.randn(1,5,5)):view(conv.weight[i]:size())
		conv.weight[i]:copy(example)
		conv.weight[i]:div(conv.weight[i]:norm())
	end
end

function SPENDenoise:rho_squared()
	local net = nn.Sequential()
	--todo: this can be replaced with a SoftPlus...
	net:add(nn.Square())
	net:add(nn.AddConstant(1.0,true))
	net:add(nn.Log())
	net:add(nn.Square())
	return net
end
function SPENDenoise:deep_spatial_filter_response(im_predict)
	local initial_conv = nnlib.SpatialConvolution(self.params.numInputChannels,self.params.energyDim,self.params.likKernelSize,self.params.likKernelSize)
	self:initializeConvWeights(initial_conv)
	local curResponse = self:energyNonlinearity()(initial_conv(im_predict))
	local depth = 2 --todo: make a command-line option for this. note: really there are depth+1 layers
	local function get_score(response)
		local score_map = nn.Squeeze()(nnlib.SpatialConvolution(self.params.energyDim,1,1,1)(response))
		local rho_score = self:rho_squared()(score_map)
		return nn.Sum(2)(nn.View(-1):setNumInputDims(2)(rho_score))
	end

	local scores = {}
	for i = 1,depth do
		table.insert(scores,get_score(curResponse))
		if(self.params.dilatedConvolution == 1) then
			local dilationScale = 3
			--NOTE: this will be slower, since it doesn't use cudnn
			local conv = nn.SpatialDilatedConvolution(self.params.energyDim,self.params.energyDim,self.params.likKernelSize,self.params.likKernelSize,1,1,0,0,dilationScale,dilationScale)(curResponse)
			curResponse = self:energyNonlinearity()(conv)
		else
			local pool = nnlib.SpatialAveragePooling(3,3,2,2)(curResponse)
			local conv = nnlib.SpatialConvolution(self.params.energyDim,self.params.energyDim,self.params.likKernelSize,self.params.likKernelSize)
			curResponse = self:energyNonlinearity()(conv(pool))
		end
	end
	table.insert(scores,get_score(curResponse))


	return nn.CAddTable()(scores)
end

function SPENDenoise:power_transform()
	local net = nn.Sequential()
	net:add(nn.Squeeze())
	net:add(nn.FFT())
	net:add(nn.Square())
	net:add(nn.Sum(4))
	return net
end

function SPENDenoise:MulWithInit(initValue)
	local m = nn.Mul()
	m.weight:fill(initValue)
	return m
end

function SPENDenoise:ExpMulWithInit(initValue)
	local m = nn.ExpMul(math.log(initValue))
	return m
end

function SPENDenoise:getLabelEnergyNetImpl()
	require 'nngraph'

	local reconstruction_weight = 1000
	local energy_terms = {}
	local learn_term_weights = true--false --todo: put back
	local input_img = nn.Identity()()
	local im_predict = nn.Identity()()

	local spatial_reconstruction_err = self:squared_loss(im_predict,input_img)
	spatial_reconstruction_err = self:ExpMulWithInit(1.0)(spatial_reconstruction_err) --the weight of the mul module gets initialized to 1
	if(self.params.noiseVariance ~= 0) then
		spatial_reconstruction_err = nn.MulConstant(1/self.params.noiseVariance,true)(spatial_reconstruction_err)
	end

	table.insert(energy_terms,spatial_reconstruction_err)

	--TODO: make this an option
	--require 'FFT'

	-- local prediction_power = self:power_transform()(im_predict)
	--todo: since we generate the 
	--local input_power = self:power_transform()(input_img)
	-- local freq_reconstruction_err = self:scaled_squared_loss(prediction_power,input_power)
	-- table.insert(energy_terms,freq_reconstruction_err)

	-- local frequency_prior = self:coordinate_scaled_sum(prediction_power)
	-- table.insert(energy_terms,frequency_prior)

	--local im_predict_scale = nn.MulConstant(255, false)(im_predict) --this is so that the scales of things is consistent with other papers
	local spatial_prior = (self.params.deepPrior == 1) and self:deep_spatial_filter_response(im_predict) or self:basic_spatial_filter_response(im_predict)
	spatial_prior = self:ExpMulWithInit(1/reconstruction_weight)(spatial_prior) --this forces the coefficients to be positive
	table.insert(energy_terms,spatial_prior)


	local total_energy = nn.CAddTable()(energy_terms) 
	self.inference_net = nn.gModule({im_predict,input_img},{total_energy})
	return self.inference_net
end

