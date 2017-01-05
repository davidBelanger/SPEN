local TrainingConfig = torch.class('TrainingConfig')

function TrainingConfig:__init(params,problem,modules,testBatcher,mode)

	self.callbacks = self:initCallbacks(params,problem,modules,testBatcher)
	self.training_options = self:initTrainingOptions(params,problem,modules)

	local criterion = problem.structured_training_loss.loss_criterion

	self.optimizer = SPENOptimizer(modules.full_net,modules.modules_to_update,criterion,self.training_options,problem) --ideally would pull this out so that we can do this with the pretraining net

	--these are to address some circular dependencies between the config objects
	self.vectorToAnalyze = self.optimizer.parameters
	self.lrd_value = self.training_options.optConfig.learningRateDecay
	self.mode = mode
end

function TrainingConfig:initTrainingOptions(params,problem,modules)
	local optInfo = OptimizationConfig:GetConfig(modules.modules_to_update,params) 
	optInfo.numEpochs = params.numEpochs
    optInfo.batchesPerEpoch = params.batchesPerEpoch
    optInfo.epochHooks = self.callbacks
    optInfo.minibatchsize = params.minibatch
    optInfo.gradientClip = params.gradientClip
	optInfo.gradientNoiseScale = params.gradientNoiseScale

    local regularization = {
        l2 = {},
        params = {}
    }

    if(params.l2 > 0) then
        table.insert(regularization.params,modules_to_update) 
        table.insert(regularization.l2,params.l2 * params.minibatch)
    end
    optInfo.regularization = regularization

	return optInfo
end

function TrainingConfig:initCallbacks(params,problem,modules,testBatcher)
	local callbacks = {}

	local experimentBase = params.name ~= "" and params.name or params.trainingMode
	os.execute('mkdir -p '..params.outDir)
	local outFileBase = params.outDir.."/"..experimentBase
	
	local saveModels = params.modelFile ~= ""
	if(saveModels) then
		 assert(params.saveFrequency > 0)
	     saver = {
		    epochHookFreq = params.saveFrequency,
		    hook = function (i)
			    --modules.full_net:clearState() --todo: ideally we'd use this, but it might not be supported by all the modules we're using
		        local file = string.format('%s-%s-%d.energy_net',params.modelFile,experimentBase,i)
		        print(string.format('saving to %s',file))
		        torch.save(file,modules.full_net)  

		       	if( (self.mode ~= "pretrainUnaries") and problem.inferencer) then
					local file = string.format('%s-%s-%d.rnn',params.modelFile,experimentBase,i)
			        print(string.format('saving to %s',file))
			        torch.save(file,problem.inferencer.rnn)  
			    end
		        local file = string.format('%s-%s-%d.optState',params.modelFile,experimentBase,i)
		       	print(string.format('saving to %s',file))
		       	torch.save(file,self.training_options)  --todo: this wouldn't work if the optimizer cloned anything in the optState, since this would point to stale values
		     end,
		     name = 'saver'
		}
		callbacks.saver = saver
	end

	self.vectorToAnalyze = nil
	local analyzer = {
		vector = self.vectorToAnalyze,
	    epochHookFreq = 1,
	    hook = function (i)
	    	if(i > 0) then
	    		if(params.printNorms) then
	    			print('NORMS: '..self.vectorToAnalyze:max().." "..self.vectorToAnalyze:min().." "..self.vectorToAnalyze:norm())
	    		end
	    		if( (self.mode ~= "pretrainUnaries") and problem.inferencer and problem.inferencer.learning_rate_weights) then
	    			print('learning rates')
	    			for _,w in ipairs(problem.inferencer.learning_rate_weights) do
	    				io.write(w[1].." ")
	    			end
	    			print('\n')
	    		end
			end
	     end,
	     name = 'analyzer'
	}
	callbacks.analyzer = analyzer

	local evaluator = {
	    epochHookFreq = params.evaluationFrequency,
	    hook = function (i)
			problem:evaluateClassifier(testBatcher, modules.full_net,i)
	     end,
	     name = 'evaluator'
	}
	callbacks.evaluator = evaluator

	--this gets set later
	self.lrd_value = nil
	local learning_rate_decay = {
		value = lrd_value,
	    epochHookFreq = 1,
	    hook = function (i)
	    	if(i > params.learningRateDecayStart) then
	    		lrd_value = nonzeroLRD
	    	else
	    		lrd_value = nonzeroLRD
	    	end
	     end,
	     name = 'lrd'
	}
	callbacks.learning_rate_decay = learning_rate_decay

	print(params.writeImageExamples)
	if(params.writeImageExamples > 0) then
		local example_generator = {
		    epochHookFreq = params.writeImageExamples,
		    hook = function (i)
		    		problem:imageAnalysis(testBatcher, modules.full_net,i)
		    end,
		    name = 'gen_examples'
		}

		callbacks.example_generator = example_generator
	end

	return callbacks

end

