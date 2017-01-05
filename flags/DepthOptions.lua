require 'torch'
package.path = package.path .. ';flags/?.lua'
require 'SPENOptions'

local cmd = torch.CmdLine()

--Architecture Options

SPENOptions:add_general_spen_options(cmd)

--data options
cmd:option('-height',160,"height of images")
cmd:option('-width',214,"width of images")
cmd:option('-energy_hid_size',25,"feature dimension")
cmd:option('-use_random_crops',0,"whether to crop the train/test images randomly to speed things up")

cmd:option('-serialize',"","where to write out the serialized options")
local params = cmd:parse(arg)

torch.save(params.serialize,params)



