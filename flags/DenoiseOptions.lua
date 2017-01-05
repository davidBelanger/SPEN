require 'torch'

local cmd = torch.CmdLine()

--todo: probably have to grab some of these from mlcoptions
cmd:option('-final_feature_dim',25,"final num channels in CNN for images")
cmd:option('-deep_features',1,"whether to use deep features")
cmd:option('-non_unary_features',1,"whether to use non-unary features")
cmd:option('-generic_convnet_energy',1,"")
cmd:option('-deep_prior',0,"whether to use deep net for prior")
cmd:option('-dilated_convolution',0,"whether to use dilated convolution, rather than conv+pool in deep prior network")
cmd:option('-noise_variance',0,"known noise variance for the data corruption process (if 0, then not used). only used for denoising problems")

cmd:option('-write_image_examples',0,"how frequently to make image examples")
cmd:option('-image_example_dir',"./ims/","base path for putting image examples")

cmd:option('-serialize',"","where to write out the serialized options")
local params = cmd:parse(arg)

torch.save(params.serialize,params)



