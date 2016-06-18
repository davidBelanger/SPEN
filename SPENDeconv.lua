local SPENDeconv,parent = torch.class('SPENDeconv','SPENProblem')

function SPENDeconv:__init(params)
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
	self.lossOnKernel = params.lossOnKernel == 1
	self.normalizedKernelEstimate = true
	self.gaussian_init = true
	self.alternatingUpdates = false

	self.useCudaForData = true
	parent.__init(self,params)
	self.params.kernelSize = 25
	self.params.padSize = 0.5*(self.params.kernelSize - 1)
	self.params.numInputChannels = 1
	self.params.numOutputChannels = 1
	self.params.numLearnedBlurChannels = 1
	self.params.likKernelSize = 7
	self.probabilityDim = 4
	self.binarylabels = true
	self.uniform_init_value = 0.5
	self.learning_rate_multiplier_per_block = {1,1}
	self.numOptimizationVariables = 2 --don't change this

	self.iterateRange = {0,1}

	self.evaluator = PSNREvaluation()
	parent.finalizeInit(self)
	if(self.lossOnKernel) then
		local crit = nn.ParallelCriterion()
		crit:add(nn.MSECriterion(),1.0) --this is for image reconstruction
		assert(self.normalizedKernelEstimate,'otherwise, do not use BCE criterion')
		crit:add(nn.MSECriterion(),self.params.kernelLossWeight)
		--crit:add(nn.BCECriterion(),self.params.kernelLossWeight)
		if(self.params.useCuda) then crit:cuda() end
		self.structured_training_loss.loss_criterion = crit
	end
	self.structured_training_loss.loss_criterion.sizeAverage = true


end

function SPENDeconv:multiPassMinibatcher(fileList,minibatch,useCuda,preprocess)
	local useCudaForData = self.useCudaForData

	return MinibatcherFromFileList(fileList,minibatch,useCudaForData,preprocess,self.shuffle)
end

function SPENDeconv:onePassMinibatcher(fileList,minibatch,useCuda,preprocess)
	local useCudaForData = self.useCudaForData
	local lazyPreprocess = not self.useCudaForData
	return OnePassMiniBatcherFromFileList(fileList,minibatch,useCudaForData,preprocess,lazyPreprocess)
end

function SPENDeconv:evaluateClassifier(testBatcher,net,iterName)
	self.evaluator:evaluateClassifier(testBatcher,net,iterName)
end


function SPENDeconv:classificationPreprocess(lab,feats,num)
	return lab,feats,num
end

function SPENDeconv:preprocess(lab,feats,num)
	if(self.useCudaForData) then
		return lab,feats,num
	else
		if(self.params.averageLoss == 1) then
			error('not implemented yet')
		end
		self.feats_cuda = self.feats_cuda or feats:cuda()
		self.feats_cuda:resize(feats:size()):copy(feats)
		if(self.lossOnKernel) then
			self.lab_cuda = self.lab_cuda or {lab[1]:cuda(), lab[2]:cuda()}
			self.lab_cuda[1]:resize(lab[1]:size()):copy(lab[1])
			self.lab_cuda[2]:resize(lab[2]:size()):copy(lab[2])
		else
			self.lab_cuda = self.lab_cuda or lab[1]:cuda()
			self.lab_cuda:resize(lab[1]:size()):copy(lab[1])
		end

		return self.lab_cuda,self.feats_cuda,num	
	end	
end

function SPENDeconv:getFeatureNetsImpl()

	self.input_features_net = nn.Identity()

	self.prediction_preprocess = nn.Identity()
	self.target_preprocess = nn.Identity() 
end

function SPENDeconv:getEnergyNetImpl()
	
end

function SPENDeconv:getProcessingNets()
	self.energy_net = nn.Sequential()
	self.energy_net_per_label = nn.Sequential()
	--todo: put back 0 initialization
	
	local init_kernel = torch.zeros(self.params.numLearnedBlurChannels,self.params.numLearnedBlurChannels,self.params.kernelSize,self.params.kernelSize)
	local center = (self.params.kernelSize - 1)/2 + 1

	if(self.gaussian_init) then
		local std = center/2 --this is just a heuristic for setting the std
		local dim = 2
		local function gaussian_density(distance_squared) return 1/math.sqrt(math.pow(2*math.pi*std,dim))*math.exp(-0.5*distance_squared/math.pow(std,2)) end
		for i = 1,self.params.kernelSize do
			for j = 1,self.params.kernelSize do
				distance_squared = (i - center)*(i - center) + (j - center)*(j - center)
				init_kernel[1][1][i][j] = gaussian_density(distance_squared)
			end
		end
		init_kernel:div(init_kernel:sum())
		if(self.normalizedKernelEstimate) then
			init_kernel:log()
		end
	else
		init_kernel:fill(-10)
		init_kernel[1][1][center][center] = 10
	end

	local init_kernel_net = nn.Constant(init_kernel,3) 
	self.input_features_probability_prediction_net = nn.ConcatTable():add(nn.Identity()):add(init_kernel_net) --this is a misnomer
	self.fixed_features_net = nn.Identity()
end

function SPENDeconv:getLabelEnergyNetImpl()
	require 'nngraph'

	local kernel_regularization_weight = 15--(25*25)
	local inv_freq_scale = 100
	local prior_nll_weight = 0.25 --this is just to set an initial scale
	local prior_alpha = 2
	local energy_terms = {}
	local input_img = nn.Identity()()
	local predictions = nn.Identity()()

	local function softmax_net()
		local net = nn.Sequential()
		net:add(nn.Reshape(self.params.kernelSize*self.params.kernelSize))
		net:add(nn.SoftMax())
		net:add(nn.Reshape(1,1,self.params.kernelSize,self.params.kernelSize))
		return net
	end



	if(not self.lossOnKernel) then
		self.prediction_selector = function(x) return nn.SelectTable(1)(x) end
	else
		self.prediction_selector = function(x) 
			local net = nn.ParallelTable()
			local ker_process = nn.Sequential():add(nn.Squeeze())
			if self.normalizedKernelEstimate then ker_process:add(softmax_net()) end
			net:add(nn.Identity()):add(ker_process)
			return net(x)
		end
 	end


	local im_predict = nn.SelectTable(1)(predictions)
	local kernel = nn.SelectTable(2)(predictions)

	if(self.normalizedKernelEstimate) then kernel = softmax_net()(kernel) end

	--kernel = nn.Print(true,false,'',function(t) print('kernel: '..t:norm()) end)(kernel)
		

	if(kernel_regularization_weight > 0) then
		local noiseVariance = 0.05
		local fft = nn.FFT()(nn.Squeeze()(kernel))
		local sq = nn.Square()(fft)
		local freq_norm = nn.MulConstant(1/noiseVariance)(nn.Sum(4)(sq))		
		local freq_inv_variances = nn.Constant(self:powerInverseVariance(),2)(freq_norm)
		freq_inv_variances = nn.MulConstant(inv_freq_scale)(freq_inv_variances)

		local logs = nn.Log()(nn.CAddTable()({freq_norm,freq_inv_variances}))
		local kernel_prior_nll = nn.Sum(2)(nn.View(-1):setNumInputDims(2)(logs))
		kernel_prior_nll = nn.Mul()(kernel_prior_nll)
		kernel_prior_nll.data.module.weight:fill(1.0)
		kernel_prior_nll = nn.MulConstant(kernel_regularization_weight)(kernel_prior_nll)


		-- local kernel_filter_responses = nn.Square()(nn.SpatialConvolution(1,self.params.energyDim,3,3,1,1)(nn.Reshape(1,self.params.kernelSize,self.params.kernelSize,true)(kernel)))
		-- local kernel_prior_nll_1 = nn.MulConstant(kernel_regularization_weight)(nn.Square()(kernel_filter_responses))
		-- local kernel_prior_nll = nn.Sum(2)(nn.View(-1):setNumInputDims(3)(kernel_prior_nll_1))
		table.insert(energy_terms,kernel_prior_nll)
	end

	local function squared_loss(x,y) 
		local diff = nn.CSubTable()({x,y})
		local sq = nn.Square()(diff)
		local rs = nn.View(-1):setNumInputDims(3)(sq) 
		return nn.Sum(2)(rs)
	end

	--note that the next line assumes the blur kernel has zero bias
	local im_blur = nn.SpatialConvolutionFromInput(self.params.numInputChannels,self.params.numOutputChannels,self.params.kernelSize,self.params.kernelSize,1,1,self.params.padSize,self.params.padSize)({im_predict,kernel})
	--im_blur = nn.Print(true,false,'',function(t) print('blur: '..t:norm()) end)(im_blur)

	local reconstruction_err = squared_loss(im_blur,input_img)
	table.insert(energy_terms,reconstruction_err)

	-- --todo: do something more for this
	local filter_responses = nn.Power(prior_alpha)(nn.SpatialConvolution(self.params.numInputChannels,self.params.energyDim,self.params.likKernelSize,self.params.likKernelSize,1,1)(im_predict))
	local reshaped_filter_responses = nn.View(-1):setNumInputDims(3)(filter_responses)
	local image_prior_nll = nn.MulConstant(prior_nll_weight)(nn.Sum(2)(reshaped_filter_responses))
	--image_prior_nll = nn.Print(false,true,'',function(t) print(t:max().." "..t:min().." "..t:mean().." "..t:sum()..'\n') end)(image_prior_nll)
	table.insert(energy_terms,image_prior_nll)


	local total_energy = nn.CAddTable()(energy_terms) 
	--total_energy = nn.PrintNoNewline(true,false,'',function(t) io.write(t[1].." ") end)(total_energy)

	self.inference_net = nn.gModule({predictions,input_img},{total_energy})
	return self.inference_net
end

function SPENDeconv:powerInverseVariance()
	local s = self.params.kernelSize


	local ivar = torch.zeros(s,s)
	local center = (s - 1)/2 + 1

	local dim = 2
	local function inv_var(distance_squared) return 1/distance_squared end
	for i = 1,self.params.kernelSize do
		for j = 1,self.params.kernelSize do
			local distance_squared = (i - center)*(i - center) + (j - center)*(j - center)
			ivar[i][j] = inv_var(distance_squared)
		end
	end
	return ivar
end

