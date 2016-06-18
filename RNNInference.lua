local RNNInference = torch.class('RNNInference')
require 'nngraph'

function RNNInference:__init(problem,params)
	----START Unpacking of Input Options from problem-----

	--the energy network E_x(y), but using pre-computed features rather than the raw value of x. 
	--This takes {labels, features} and returns a single number per minibatch element--the energy network E_x(y), 
	--but using pre-computed features rather than the raw value of x. This takes {labels, features} and returns a single number per minibatch element
	local inference_net = problem.inference_net 
	

	--Feature mapping F(x). This may be pretrained using classification, or loaded from file. 
	--If the training mode is 'clampFeatures', then we don't update its parameters, and don't even backprop through it during training. 
	local features_net = problem.fixed_features_net 

	--(Optional) The overall feature mapping is fixed_features_net followed by learned_features_net. These features are learned even in 'clampFeatures' mode.
	local learned_features_net = problem.learned_features_net 

	--This network takes {x,F(x)} and returns an initial guess y_0 for the labels. 
	--The reason it takes the raw input x is that this might be important for getting the size of the inputs when the problem can have variable-sized inputs.
	--See SPENProblem.lua to see how this interacts with the --initAtLocalPrediction flag. 
	--Generally, you don't implement initialization_net directly. Instead, you decide to init the labels with the outputs of a local classifier, 
	--or initialize them to some fixed hard-coded value (eg. 0).
	local init_net = problem.initialization_net 

	--Everything is set up for unconstrained optimization. This maps things onto the constrain set at the end of optimization. For example, it converts logits to probabilities. Set to Identity() if you don't need a transformation. 
	local iterate_transform = problem.iterate_transform 
	

	--For some problems, there are multiple optimization variables, but downstream you only care about one of them. 
	--For example, in blind deconvolution we have a latent image and latent blur kernel and we often only care about the image. 
	--We provide limited support for this. Depending on the problem structure, you may want to do something different than joint gradient descent on all variables, eg. block coordinate descent. 
	--Here, we assume that y is a table of optimization variables.
	local prediction_selector = problem.prediction_selector --For some problems, such as blind deconvolution, there are multiple optimization variables, but downstream you only care about one of them. This grabs that one from a table of optimization variables. SPENProblem has a default Identity implementation. 
	self.learning_rate_multiplier_per_block = problem.learning_rate_multiplier_per_block --Table of learning rates to use for table element of y. 
	self.numOptimizationVariables = problem.numOptimizationVariables --Size of the y table. 
	self.alternatingUpdates = problem.alternatingUpdates --Whether to round-robin gradient descent on each element of y, rather than stepping on all of them at once. Basically, poor-man's block coordinate descent. 

	----END Unpacking of Input Options from problem-----

	----START Unpacking of Input Options from params-----
	self.cuda = params.useCuda

	self.numIters = params.maxInferenceIters
	self.numLabels = params.numClasses
	self.numFeatures = params.vocabSize
	self.predictor_net = predictor_net

	local averageLoss = params.averageLoss == 1
	local unconstrainedIterates = params.unconstrainedIterates == 1
	local untiedEnergyNets = params.untiedEnergyNets == 1
	local clonePredictor = params.clonePredictor  == 1
	local learnHyperparams = params.learnInferenceHyperparams == 1
	local useMomentum = params.inferenceMomentum > 0
	local gamma = params.inferenceMomentum
	local learning_rate_scale = params.inferenceLearningRate
	local learning_rate_decay = params.inferenceLearningRateDecay
	local learning_rate_power = params.inferenceLearningRatePower
	----END Unpacking of Input Options from params-----




	self.features_net = features_net
	self.predictionSize = torch.LongStorage({self.numLabels})
    self.energy_net = nn.Sequential():add(nn.ParallelTable():add(iterate_transform()):add(nn.Identity())):add(inference_net)

	local hidStateDimension = useMomentum and self.numLabels or 1 
	local returnAllIterates = averageLoss
	local hyperparams = 
	{
		learning_rate_scale = learning_rate_scale,
		learning_rate = 0, --this gets overwritten below
		useMomentum = useMomentum, 
		gamma = gamma, 
		temperature = 1.0,
		learnHyperparams = learnHyperparams,
		doProjection = not unconstrainedIterates,
		finiteDifferenceStep = params.finiteDifferenceStep,
		iterateRange = iterateRange
	}

	local features_placeholder = nn.Identity()()
	local init_labels_placeholder = nn.Identity()()
	local curStep = {
		features = features_placeholder,
		y = init_labels_placeholder,
		h = init_hid_state
	}

	local iterates = {}
	local function transform_to_prob(y) if(unconstrainedIterates and iterate_transform) then return iterate_transform()(y) else return y end end
	function store_iterate(y) table.insert(iterates,transform_to_prob(y)) end
	
	self.parameters_container_without_features = {}

	for i = 1,self.numIters 
 	do
		local hyperparams_for_timestep = Util:copyTable(hyperparams)


		hyperparams.learning_rate = hyperparams.learning_rate_scale/math.pow(1 + learning_rate_decay*i,learning_rate_power) 
		hyperparams.timestep = i
		if(returnAllIterates) then store_iterate(curStep.y) end

		local cl
		if(not untiedEnergyNets) then
				cl = self.energy_net
		else
			cl = (i == 1) and self.energy_net or self.energy_net:clone()
		end
		
		curStep = self:getRecurrentCell(curStep,cl,hyperparams) 

	end

	if(returnAllIterates) then store_iterate(curStep.y) end

	local final_prediction = returnAllIterates and nn.Identity()(iterates) or transform_to_prob(prediction_selector(curStep.y))
	local rnn_without_features = nn.gModule({features_placeholder,init_labels_placeholder},{final_prediction})
	
	rnn_without_features.name = 'rnn'
	table.insert(self.parameters_container_without_features,rnn_without_features) 

	local raw_features = nn.Identity()()
	self.featureWrapperModule = nn.TruncatedBackprop()
	self.featureWrapperModule:add(features_net)
	
	local features
	if(learned_features_net) then
		features = nn.Sequential():add(self.featureWrapperModule):add(problem.learned_features_net)(raw_features)
	else
		features = self.featureWrapperModule(raw_features)
	end

	local init_hid_state = nil

	local init_labels_net = init_net({raw_features,features})

	local features_and_labels_init = nn.gModule({raw_features},{features,init_labels_net})

	self.rnn = nn.Sequential():add(features_and_labels_init):add(rnn_without_features)
	if(self.cuda) then 
		self.rnn:cuda()
		features_net:cuda()
	end
	self.parameters_container = Util:copyTable(self.parameters_container_without_features) --shallow copy
	table.insert(self.parameters_container,self.features_net)
	if(self.learning_rate_modules) then
		self.learning_rate_weights = {}
		for _,m in ipairs(self.learning_rate_modules) do
			table.insert(self.learning_rate_weights,m.data.module.weight)
		end
	end
end



--this takes {features_t,{y_t,h_t}} and returns {y_{t+1},h_{t+1}}
function RNNInference:getRecurrentCell(inputs,energy_network,hyperparams)
	assert(not (hyperparams.useMomentum and hyperparams.useRDA))
	local features = inputs.features
	local yt = inputs.y
	local ht = inputs.h

	local grad_t = nn.GradientDirection(energy_network,1,hyperparams.finiteDifferenceStep,false)({yt,features})

	local ht1, direction
	if(hyperparams.useMomentum) then
		--ht1 = gamma*ht + (1- gamma)*grad_t
		if(hyperparams.timestep > 1) then
			local term1 = nn.MulConstant(hyperparams.gamma,true)(ht)
			local term2 = nn.MulConstant(1 - hyperparams.gamma,true)(grad_t)
			ht1 = nn.CAddTable()({term1,term2})
		else
			ht1 = grad_t
		end
		direction = ht1
	else
		ht1 = ht
		direction = grad_t
	end

	local function project(x) return x end
	if(hyperparams.doProjection) then
		function project(x) 
			assert(hyperparams.iterateRange)
			return nn.Clamp(unpack(hyperparams.iterateRange))(x)
		end
	end

	local function noop_update(point,direction)
		return point
	end

	local function gradient_update(point,direction,lr,constant_lr_scale) 
       local scaled_direction 
       if(hyperparams.learnHyperparams) then
           scaled_direction = nn.Mul()(direction)
           local weight = scaled_direction.data.module.weight
           weight:fill(-lr)
           self.learning_rate_modules = self.learning_rate_modules or {}
           table.insert(self.learning_rate_modules,scaled_direction)
           if(constant_lr_scale) then
	           scaled_direction = nn.MulConstant(constant_lr_scale,true)(scaled_direction)
	        end
       else
       	   constant_lr_scale = constant_lr_scale or 1.0
           scaled_direction = nn.MulConstant(-lr*constant_lr_scale,true)(direction)
        end

        return nn.CAddTable()({point,scaled_direction})
	end

	local yt1

	if(self.numOptimizationVariables > 1) then
		yt1 = {}
		for i = 1,self.numOptimizationVariables do
			local yti = nn.SelectTable(i)(yt)

		   local step
           if((not self.alternatingUpdates) or (hyperparams.timestep % i == 0)) then 
           		local di  = nn.SelectTable(i)(direction)
                step = project(gradient_update(yti,di,hyperparams.learning_rate,self.learning_rate_multiplier_per_block[i]))
           else
                step = noop_update(yti,nil)
           end
			table.insert(yt1,step)
		end
		yt1 = nn.Identity()(yt1)
	else
        yt1 = project(gradient_update(yt,direction,hyperparams.learning_rate))
	end

	local to_return = 
	{
		features = features,
		y = yt1,
		h = ht1
	}
	return to_return

end


function RNNInference:setFeatureBackprop(value)
	self.featureWrapperModule.doBackprop = value
end


function RNNInference:training()
	self.trainMode = true
	for _,net in ipairs({self.features_net,self.rnn}) do 
		net:training()
	end
end

function RNNInference:evaluate()
	self.trainMode = false
	for _,net in ipairs({self.features_net,self.rnn}) do 
		net:evaluate()
	end
end


--Above, most things are tailored to unconstrained optimization problems. In particular, if iterates are constrained
--to be on the simplex, we use logits as our optimization variables, and a sigmoid/softmax is the first layer of the energy network.
--however, we could do optimization directly over the simplex, using entropic mirror descent (emd). Below is an efficient implementation for bernouli probabilities. 
--We don't use this in practice because the sigmoids lead to lots of vanishing gradient issues. 
--When we work in logspace, the computation graph looks more like a GRU/LSTM in that it has additive updates

--returns a module that inputs {y,g} and outputs the result of taking an EMD step in -g direction on y
function RNNInference:emd(y,direction,learning_rate, temperature,learnHyperparams)
	local epsilon = 0.000001 --this is used to prevent dividing by zero
	
	--log(x/(1-x))
	local one_minus_y = nn.AddConstant(1 + epsilon,true)(nn.MulConstant(-1,true)(y))
	local ratio = nn.CDivTable()({y,one_minus_y})
	local log_ratio = nn.Log()(nn.AddConstant(epsilon,true)(ratio))

	-- -2 lambda g
	local scaled_grad 
	if(learnHyperparams) then
		scaled_grad = nn.Mul()(direction)
		scaled_grad.weight:fill(-2*learning_rate)
	else
		scaled_grad = nn.MulConstant(-2*learning_rate,true)(direction)
	end

	local logit_step = nn.CAddTable()({log_ratio,scaled_grad})

	local logit_step_with_temp
	if(learnHyperparams) then
		logit_step_with_temp = nn.Mul(logit_step) --todo: BAD: this learns a separate temp for every coordinate
		temp.weight:fill(1/temperature)
	elseif(temperature ~= 1.0) then
		logit_step_with_temp = nn.MulConstant(1/temperature,true)(logit_step)
	else
		logit_step_with_temp = logit_step
	end

	return nn.Sigmoid()(logit_step_with_temp)
end

--this is a useful utility function for analyzing how peaked the outputs are. Use this for problems where we we have labels in {0,1}, but we relax to [0,1]. 
function RNNInference:peakedness(probs,inputs_are_logits)
	if(inputs_are_logits and not self.sigmoid_transform) then
		self.sigmoid_transform = nn.Sigmoid()
		self.sigmoid_transform:type(probs:type())
	end
	if(inputs_are_logits) then
		probs = self.sigmoid_transform:forward(logits)  
	end

	self.oneMinusProb = self.oneMinusProb or probs:clone()
	self.oneMinusProb:resizeAs(probs):copy(probs):mul(-1):add(1)
	self.oneMinusProb:cmin(probs)

	return 1-self.oneMinusProb:mean()
end




