-- local OptimizationConfig = torch.class('OptimizationConfig')
-- require 'optim'
-- function OptimizationConfig:GetConfig(training_net,params)
--     local optInfo = {}
--     local util = require 'model_utils'



--     if(params.initOptState ~= '')then
--         print(string.format('initializing optimizer from %s',params.initOptState))   
--         optInfo = torch.load(params.initOptState)
--     else
--         if(params.optimMethod == "adagrad") then
--             print('using adagrad')
--             local initVar = 1.0
--             local len = util.num_parameters(modules_to_update)
--             local example = modules_to_update[1]:parameters()[1][1]
--             assert(example)
--             local paramVariance = example.new():resize(len):fill(initVar)
--             local paramStd = example.new():resize(len):fill(math.sqrt(initVar))
--             optInfo = {
--                 optimMethod = optim.adagrad,
--                 optConfig = {
--                     learningRate = params.learningRate,
--                     learningRateDecay = params.learningRateDecay,
--                 },   
--                 optState = {
--                     initVar = initVar,
--                     paramVariance = paramVariance,
--                     paramStd = paramStd 
--                 }
--             }
--         elseif(params.optimMethod == "sgd") then
--             print('using SGD')
--             local useMomentum = 0.9
--             local dampening = 0
--             local nesterov = false
--             optInfo = {
--               optimMethod = optim.sgd,
--               optConfig = {
--                 learningRate = params.learningRate,
--                 learningRateDecay = params.learningRateDecay,
--                 momentum = useMomentum,
--                 dampening = dampening,
--                 nesterov = nesterov,
--             },
--               optState = {}   
--             }
--         elseif(params.optimMethod == "adam") then 
--             print('using adam')
--             optInfo = {
--               optimMethod = optim.adam,
--               optConfig = {
--                 learningRate = params.learningRate,
--                 beta1 = params.adamBeta1,
--                 beta2 =  params.adamBeta2,
--                 epsilon = params.adamEpsilon,
--                 --lambda =  1-1e-8,
--             },
--               optState = {}   
--             }
--         else
--             assert(false,'invalid optim Method specified')
--         end
--     end



--     optInfo.regularization = params.regularization
--     optInfo.gradientClip = params.gradientClip
--     return optInfo

-- end

