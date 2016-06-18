require 'torch'

local cmd = torch.CmdLine()

cmd:option('-name',"","")
cmd:option('-learningRate',0.1,'')
cmd:option('-learningRateDecay',0,'lr = init_lr/(1 + lrd*n_evals)')
cmd:option('-learningRateDecayStart',25,'when to start decaying lr')
cmd:option('-l2',0,'l2 regularization')
cmd:option('-evaluationFrequency',25,'evaluation frequency')
cmd:option('-saveFrequency',-1,"how often to save the model")
cmd:option('-initParams','','Model to initialize params from. Assuming this is the same structure as the training_net allocated in this code')
cmd:option('-numEpochs',0,"how many training epochs when the feature layer is clamped")
cmd:option('-batchesPerEpoch',25,"how many batches per epoch in structured training")

cmd:option('-optimMethod',"sgd",'learning optimization method')
cmd:option('-adamEpsilon',1e-8,"")
cmd:option('-adamBeta1',0.9,"")
cmd:option('-adamBeta2',0.999,"")
cmd:option('-gradientClip',0,"norm bound on gradient")
cmd:option('-gradientNoiseScale',0,"")
cmd:option('-trainingMode',"","(pretrainUnaries,clampFeatures,updateAll)")

cmd:option('-learnUnaryInFirstPass',0,"whether to optimize the linear predictor of the unaries in the first learning pass. Only used in clampFeatures mode")

cmd:option('-initOptState',"","where to load optimization config from")

cmd:option('-serialize',"","where to write out the serialized options")
local params = cmd:parse(arg)

torch.save(params.serialize,params)



