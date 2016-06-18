local Optimizer = torch.class('Optimizer')


--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015


function Optimizer:__init(model,modules_to_update,criterion, config)
    local model_utils = require 'model_utils'
    assert(config)
    self.config = config

    self.model = model
    self.optState = config.optState	
    self.optConfig = config.optConfig
    self.optimMethod = config.optimMethod
    self.regularization = config.regularization
    self.minibatchsize = config.minibatchsize


    self.totalError = torch.Tensor(1):zero()
    self.useGradientNoise = config.gradientNoiseScale and config.gradientNoiseScale > 0
    self.gradientNoiseScale = config.gradientNoiseScale

    self.gradBound = config.gradientClip
    self.clampGradient = config.gradientClip > 0 


    local parameters
    local gradParameters


    if(not Util:isArray(modules_to_update)) then
        parameters, gradParameters = modules_to_update:getParameters() 
    else
        local cont = nn.Container()
        for _, m in pairs(modules_to_update) do
            cont:add(m)
        end
        parameters, gradParameters = cont:getParameters()
--        parameters, gradParameters =  model_utils.combine_all_parameters(unpack(modules_to_update)) --
    end
    self.parameters = parameters
    assert(self.parameters:nElement() > 0)
    print('optimizing '..self.parameters:nElement().." parameters")

    self.gradParameters = gradParameters

    self.l2s = {}
    self.params = {}
    self.grads = {}
    for i = 1,#self.regularization.params do
            local params,grad = self.regularization.params[i]:parameters()
            local l2 = self.regularization.l2[i]
            table.insert(self.params,params)
            table.insert(self.grads,grad)
            table.insert(self.l2s,l2)
    end
    self.numRegularizers = #self.l2s
    self.numIters = 0

    self.totalError:typeAs(parameters)

     self.criterion = criterion
    for hookIdx = 1,#self.config.epochHooks do
        local hook = self.config.epochHooks[hookIdx]
        if( hook.epochHookFreq == 1) then
            hook.hook(0)
        end
    end
end

function Optimizer:train(batchSampler)
	 local prevTime = sys.clock()
     local batchesPerEpoch = self.config.batchesPerEpoch
     local tst_lab,tst_data = batchSampler()
     local epochSize = batchesPerEpoch*self.minibatchsize
     local numProcessed = 0
     
    local i = 0
    while i < self.config.numEpochs  do
        self.totalError:zero()
        i = i + 1
        self.numIters = i
        for j = 1,batchesPerEpoch do
    	    local minibatch_targets,minibatch_inputs = batchSampler()
            --in some cases, the targets are actually part of the inputs with some weird table structure. Need to account for this.
            numProcessed = numProcessed + self.minibatchsize
            self:trainBatch(minibatch_inputs,minibatch_targets) 
        end

        local avgError = self.totalError[1]/batchesPerEpoch
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        local rate = numProcessed/ElapsedTime
        numProcessed = 0
        prevTime = currTime
        print(string.format('\nIter: %d\navg loss in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, ElapsedTime,ElapsedTime/batchesPerEpoch))
        --print(string.format('cur learning rate = %f',self.optConfig.learningRate))
        print(string.format('examples/sec = %f',rate))
        self:postEpoch()

         for _,hook in pairs(self.config.epochHooks) do
            if( i % hook.epochHookFreq == 0) then
                hook.hook(i)
            end
       end

    end
end

function Optimizer:postEpoch()
    --this is to be overriden by children of Optimizer
end

function  Optimizer:preBatch()
    --optional abstract method to be implemented by children
end

require 'image' --todo: remove
function Optimizer:trainBatch(inputs, targets)
    assert(inputs)
    assert(targets)

    local parameters = self.parameters
    local gradParameters = self.gradParameters

    self:preBatch()

    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.model:zeroGradParameters()
        --print('FORWARD')
        local output = self.model:forward(inputs)

        local err = self.criterion:forward(output, targets)
        local df_do = self.criterion:backward(output, targets)

        self.model:backward(inputs, df_do) 
        --note we don't bother adding regularizer to the objective calculation. 
        for i = 1,self.numRegularizers do
            local l2 = self.l2s[i]
            for j = 1,#self.params[i] do
                self.grads[i][j]:add(l2,self.params[i][j])
            end
        end

        --todo: scale this by the current norm of the params or something?
        if(self.useGradientNoise) then
            self.noise = self.noise or gradParameters:clone()
            torch.randn(self.noise,self.noise:size())
            local scale = self.gradientNoiseScale/math.sqrt(self.numIters)
            gradParameters:add(scale,self.noise)
        end

        if(self.clampGradient) then
            local norm = gradParameters:norm()
            if(norm > self.gradBound) then
                gradParameters:mul(self.gradBound/norm)
            end
        end

        self.totalError[1] = self.totalError[1] + err
        return err, gradParameters
    end
    self.optimMethod(fEval, parameters, self.optConfig, self.optState)
    if(self.assert) then assert(parameters:eq(parameters):all(),'NANs in parameters after gradient step') end
    return err
end


