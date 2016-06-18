local seed = 12345
torch.manualSeed(seed)

require 'nn'
require 'RNNInference'
package.path = package.path .. ';../../torch-util/?.lua'
package.path = package.path .. ';../?.lua'

require 'emd'
require 'MySGD'

local lr = 0.1
local gamma = 0
local temp = 1

local emd_network = RNNInference:emd(lr,temp,false)


local optConfig = {
	learningRate = lr,
	momentum = gamma,
	nesterov = false,
	learningRateDecay = 0,
	useEMD = true,
	extraEntropyWeight = 0,
}

local optState = {}

local x = torch.randn(64,50)

local function setParams(params)
	params:abs():cmax(0.001):cmin(0.999)
end

local model_gt = nn.Sequential():add(nn.Linear(50,25))
local params,gradParams = model_gt:getParameters()
setParams(params)
local y = model_gt:forward(x):clone()
y:add(0.25,torch.randn(y:size()))
model_gt = nil

local model = nn.Sequential():add(nn.Linear(50,25))
local params,gradParams = model:getParameters()
setParams(params)

local model2 = model:clone()
local params2,gradParams2 = model2:getParameters()

local criterion = nn.MSECriterion()

local curErr = 0
local function opfunc(m,params,gradParams,stats)
	local yhat = m:forward(x)
	local err = criterion:forward(yhat,y)
	local bg = criterion:backward(yhat,y)
	m:backward(x,bg)
	stats.curErr = err
	return err, gradParams
end
local stats1 = {curErr = 0}
local stats2 = {curErr = 0}
local function opfunc1(params) return opfunc(model,params,gradParams,stats1) end
local function opfunc2(params) return opfunc(model2,params,gradParams2,stats2) end


local sgd = MySGD(optConfig)
local before = params:clone()
local after = params:clone()
for t = 1,2 do
	model:zeroGradParameters()
	model2:zeroGradParameters()
	before:copy(params2)
	sgd:step(opfunc2,params2,optConfig,optState)

	--optim.emd(opfunc1,params,optConfig,optState)
	local emd_out = emd_network:forward({before,gradParams2})
	after:copy(params2)
	local diff = emd_out  - after
	local ratio = emd_out:clone():cdiv(after)
	print(ratio:min().." "..ratio:max())
	-- print(diff:norm())
	-- print(emd_out:norm())
	print('-----')
	-- if(t % 10 == 0) then print("1: "..stats1.curErr) end
	-- if(t % 10 == 0) then print("2: "..stats2.curErr.."\n") end

end


