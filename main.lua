local seed = 12345
torch.manualSeed(seed)

--these are general torch dependcies 
--depending on the command line options, we may also require cunn and cudnn later on
require 'nn'
require 'optim'

--these files are from the torch-util project https://github.com/davidBelanger/torch-util
package.path = package.path .. ';../torch-util/?.lua'
require 'MinibatcherFromFile'
require 'MinibatcherFromFileList'
require 'OnePassMiniBatcherFromFileList'
require 'Util'
require 'Assert'
require 'Print'
model_utils = require 'model_utils'

--these are dependencies that are in this project
require 'TruncatedBackprop'
require 'OptimizationConfig'
require 'SPENProblem'
require 'SPENMultiLabelClassification'
require 'SPENDenoise'
require 'SingleBatcher'
require 'ImageAnalysis'
require 'RepeatedCriterion'


require 'EpochDropout'
require 'Optimizer'
require 'SPENOptimizer'
require 'PSNREvaluation'
require 'ExpMul'

require 'MultiLabelEvaluation'
require 'ScaledBCECriterion'
require 'ScaledMSECriterion'

require 'RNNInference'
require 'GradientDirection'
require 'PrintNoNewline'
require 'Constant'
require 'TrainingConfig'

local cmd = torch.CmdLine()

--NOTE: Additional Options are in TrainingOptions.lua. That file contains ones that specific to each stage of training. 

--General Options
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn',false,'whether to use cudnn, where appropriate')
cmd:option('-minibatch',32,'minibatch size')
cmd:option('-profile',0,"whether to do profiling")
cmd:option('-shuffle',1,'whether to shuffle the data after you take a pass through it')
cmd:option('-testOnly',0,"whether to only run evaluation")
cmd:option('-problem','ML',"what kind of problem (ML,Denoise)")
cmd:option('-trainingConfigs',"","list of serialized training config files")

--Data Options
cmd:option('-trainList','','list of torch format train files')
cmd:option('-testList','','list of torch format dev/test files')

--Pretrained Parameters Options
cmd:option('-initFeaturesNet',"","where to load pretrained feature network from")
cmd:option('-initFullNet',"","where to load pretrained parameters for full network from")
cmd:option('-initEnergy',"","where to load energy net from. note this is just for inference_net and does not load the features")

--Architecture Options
cmd:option('-directEnergy',1,"whether to use unary energy terms")
cmd:option('-linearFeatures',0,'whether to use linear features')
cmd:option('-embeddingDim',-1,'embedding dimension')
cmd:option('-featureDim',-1,'dimensionality of feature maps')
cmd:option('-energyDim',-1,'dimensionality of feature maps in energy')
cmd:option('-dropout',0,"droupout rate")
cmd:option('-featuresNonlinearity',"ReLU","what kind of nonlinearity to use")
cmd:option('-energyNonlinearity',"SoftPlus","what kind of nonlinearity to use. Don't use ReLU if you want the finite differences to behave reasonably.")
cmd:option('-initUnaryWeight',1.0,'multiplier on the unary term')

--Inference Options
cmd:option('-binaryInference',1,"whether the prediction variables are {0,1}")
cmd:option('-inferenceLearningRate',0.1,"learning rate for inference")
cmd:option('-inferenceLearningRatePower',1.0,"learning rate power for inference")

cmd:option('-maxInferenceIters',30,"how many inference iters to perform")
cmd:option('-project',0,'whether to condition on a pretrained initial embedding layer (to speed up feature computation during learning)')
cmd:option('-clonePredictor',0,'whether to use a fixed predictor net, or one that is learned')
cmd:option('-learnInferenceHyperparams',1,'whether to learn the inference hyperparams (eg. learning rates)')
cmd:option('-inferenceMomentum',0,"") 
cmd:option('-inferenceLearningRateDecay',0,"not used if learnInferenceHyperparams is true") 
cmd:option('-inferencePerturbation',0,"whether to add random perturbations to inference optimization during training. currently not fully supported.")
cmd:option('-scaleDirectEnergy',0,"when clamping the unaries, whether to learn a global scale on them")
cmd:option('-finiteDifferenceStep',0.0001," epsilon used in finite difference approximation of Hessian-vector products")
cmd:option('-unconstrainedIterates',1,"whether to use logits as inference iterates") --TODO: remove?
cmd:option('-initAtLocalPrediction',0,"whether to init prediction using unary predictor")
cmd:option('-untiedEnergyNets',0,"whether to use untied energy nets at each inference timestep (introduces lots more parameters)")

--Loss Options
cmd:option('-lossType',"log","what training loss to use")
cmd:option('-positiveWeight',1.0,"how much weight to place on positive class examples (to give recall bias)")
cmd:option('-averageLoss',0,"whether to use sum_t Loss(y_t,t) rather than just L(y_T,y). Might help things converge faster.")
cmd:option('-interpCriterion',0,"only used with averageLoss mode. Whether to define the ground truth loss for iterate t to be an interpolation between the ground truth and y_0.")

--Evaluation Options
cmd:option('-evaluateByRanking',0,"only used for multi-label classification")
cmd:option('-predictionThresh',-1,'threshold for multi-label prediction')
cmd:option('-resultsFile',"","optional file base name for writing results files")
cmd:option('-testMinibatchSize',6400,"batch size at test time")
cmd:option('-modelFile',"","base name for where to save models. the output .rnn file contains the full unrolled inference network. The .energy_net file only contains the energy network.")
cmd:option('-outDir',"./results/","base name for where to save models")
cmd:option('-writeImageExamples',0,"how frequently to make image examples")
cmd:option('-imageExampleDir',"./ims/","base path for putting image examples")
cmd:option('-printNorms',false,"whether to print the various norms of the parameters after every epoch")

--Image Specific Options
cmd:option('-finalFeatureDim',25,"final num channels in CNN for images")
cmd:option('-deepFeatures',1,"whether to use deep features")
cmd:option('-nonUnaryFeatures',1,"whether to use non-unary features")
cmd:option('-genericConvnetEnergy',1,"")
cmd:option('-deepPrior',0,"whether to use deep net for prior")
cmd:option('-dilatedConvolution',0,"whether to use dilated convolution, rather than conv+pool in deep prior network")
cmd:option('-noiseVariance',0,"known noise variance for the data corruption process (if 0, then not used). only used for denoising problems")

--Multi-Label Classification Options
cmd:option('-inputSize',"",'dimensionality of the inputs')
cmd:option('-labelDim',-1,'dimensionality of the labels')
cmd:option('-labelEnergy',0,"whether to use label energy")
cmd:option('-conditionalLabelEnergy',0,"whether to use conditional label energy. usually don't want to use both this and the labelEnergy")
cmd:option('-labelEnergyType',"deep","parametrization for label energy net")

--Deep Mean Field Options (only for Multi-label Classification)
cmd:option('-meanField',0,"whether to use mean field network for classification")
cmd:option('-meanFieldIters',5,"number of MF iters")
cmd:option('-dataIndependentMeanField',0,"whether pairwise potentials should be data independent")


local params = cmd:parse(arg)

if(params.profile == 1) then
   require 'Pepperfish'  
   profiler = newProfiler()
   profiler:start()
end

print(params)

local useCuda = params.gpuid >= 0
params.useCuda = useCuda
if(useCuda)then
    print('USING GPU '..params.gpuid)
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(params.gpuid + 1) 
    cutorch.manualSeed(seed)
end

nnlib = nil
if(params.cudnn) then
	require 'cudnn'
	nnlib = cudnn
end

--TODO: remove some of these
local problem
if(params.problem == "ML") then
  problem = SPENMultiLabelClassification(params)
elseif(params.problem == "Image") then
	problem = SPENImage(params)
elseif(params.problem == "Denoise") then
	torch.setdefaulttensortype('torch.FloatTensor')
	problem = SPENDenoise(params)
elseif(params.problem == "Deconv") then
	torch.setdefaulttensortype('torch.FloatTensor')
	problem = SPENDeconv(params)	
elseif(params.problem == "Pose") then
	require 'PoseEvaluation'
	require 'cudnn'
	torch.setdefaulttensortype('torch.FloatTensor')
	problem = SPENPose(params)	
end

local batcher = problem:minibatcher(params.trainList,false)
local testBatcher = problem:minibatcher(params.testList,true)
local criterion = problem.structured_training_loss.loss_criterion 

local inferencer = RNNInference(problem,params)

problem.inferencer = inferencer --this is network API equivalent to a simple feed-forward classifier. It is the unrolled computation graph for doing gradient descent on the energy function
local testInferencer = inferencer --this doesn't necessarily have to be the same as the one used at train time. For example, we may want to run for more gradient descent iterations

if(params.initFullNet ~= "") then
	print('initializing parameters from '..params.initFullNet)
	inferencer.rnn:getParameters():copy(torch.load(params.initFullNet):getParameters())
end

for paramsFile in io.lines(params.trainingConfigs) do

	print('loading specific training config from '..paramsFile)
	local specific_params = torch.load(paramsFile)
	if(specific_params.numEpochs > 0) then 
		local all_params = Util:copyTable(params)
		for k,v in pairs(specific_params) do 
			assert(not all_params[k],'repeated key')
			all_params[k] = v 
		end

		local modules
		local mode = all_params.trainingMode
		if(mode == "pretrainUnaries") then
			modules = {
				modules_to_update = problem.input_features_pretraining_net,
				full_net = problem.input_features_pretraining_net
			}
		elseif(mode == "clampFeatures") then
			modules = {
				modules_to_update = inferencer.parameters_container_without_features,
				full_net = inferencer.rnn
			}
			inferencer:setFeatureBackprop(false)
		elseif(mode == "updateAll") then
			modules = {
				modules_to_update = inferencer.parameters_container,
				full_net = inferencer.rnn
			}
			inferencer:setFeatureBackprop(true)
		else
			error('invalid training mode: '..mode)
		end

		local config = TrainingConfig(all_params,problem,modules,testBatcher,mode)
		if(params.testOnly == 1) then
			config.callbacks.evaluator.hook(0)
			break
		end

		print(specific_params)

		config.optimizer:train(function () return batcher:getBatch() end)
	end
end


if(params.profile == 1) then
	profiler:stop()
	local report = "profile.txt"

	print('writing profiling report to: '..report)
    local outfile = io.open( report, "w+" )
    profiler:report( outfile )
    outfile:close()
end



