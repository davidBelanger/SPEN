require 'torch'

local cmd = torch.CmdLine()

cmd:option('-name',"","")
cmd:option('-learning_rate',0.1,'')
cmd:option('-learning_rate_decay',0,'lr = init_lr/(1 + lrd*n_evals)')
cmd:option('-learning_rate_decay_start',25,'when to start decaying lr')
cmd:option('-l2',0,'l2 regularization')
cmd:option('-evaluation_frequency',25,'evaluation frequency')
cmd:option('-save_frequency',-1,"how often to save the model")
cmd:option('-init_params','','Model to initialize params from. Assuming this is the same structure as the training_net allocated in this code')
cmd:option('-num_epochs',0,"how many training epochs when the feature layer is clamped")
cmd:option('-batches_per_epoch',25,"how many batches per epoch in structured training")

cmd:option('-optim_method',"sgd",'learning optimization method')
cmd:option('-adam_epsilon',1e-8,"")
cmd:option('-adam_beta1',0.9,"")
cmd:option('-adam_beta2',0.999,"")
cmd:option('-gradient_clip',0,"norm bound on gradient")
cmd:option('-gradient_noise_scale',0,"")
cmd:option('-training_mode',"","(pretrain_unaries,clamp_features,clamp_unaries,update_all)")

--cmd:option('-learn_unary_in_first_pass',0,"whether to optimize the linear predictor of the unaries in the first learning pass. Only used in clamp_features mode")
cmd:option('-init_opt_state',"","where to load optimization config from")

cmd:option('-serialize',"","where to write out the serialized options")
local params = cmd:parse(arg)

torch.save(params.serialize,params)



