require 'torch'
package.path = package.path .. ';flags/?.lua'
require 'SPENOptions'

local cmd = torch.CmdLine()

--Architecture Options

SPENOptions:add_general_spen_options(cmd)

--data options
cmd:option('-length',10,"length of the sequences")
cmd:option('-domain_size',5,"domain size of categorical tags")
cmd:option('-feature_size',6,"dimensionality of per-token features")

--architecture options
cmd:option('-feature_width',3,"width of convolutions in feature network")
cmd:option('-feature_hid_size',5,"dimension of per-token features from feature network")
cmd:option('-energy_hid_size',4,"dimension of hidden layers in energy network")

cmd:option('-conditional_label_energy',1,"whether the label energy should depend on the features")

cmd:option('-features_nonlinearity',"ReLU","what kind of nonlinearity to use")
cmd:option('-energy_nonlinearity',"SoftPlus","what kind of nonlinearity to use. Don't use ReLU if you want the finite differences to behave reasonably.")
cmd:option('-dropout',0,"droupout rate")

cmd:option('-serialize',"","where to write out the serialized options")
local params = cmd:parse(arg)

torch.save(params.serialize,params)



