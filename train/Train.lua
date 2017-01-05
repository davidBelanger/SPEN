local Train = torch.class('Train')


--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015


function Train:__init(loss_wrapper, batcher, optimization_config, general_config, callbacks)
    assert(config)
    self.optimization_config = optimization_config
    self.general_config = general_config
    self.callbacks = callbacks or {}
    self.loss_wrapper = loss_wrapper
    self.batch_iterator = batcher:get_ongoing_iterator()

    self.model = optimization_config.modules_to_update

    self.parameters, self.grad_parameters = self.model:getParameters()
    self.data_for_callbacks = {
        parameters = self.parameters,
        grad_parameters = self.grad_parameters
    }
end

function Train:train()
	 local prev_time = sys.clock()
     local batches_per_epoch = self.general_config.batches_per_epoch
     local epoch_size = batches_per_epoch*self.general_config.batch_size
     local num_processed = 0
     
    local i = 0
    self.total_error = 0
    while i < self.general_config.num_epochs  do
        self.total_error = 0
        i = i + 1
        self.num_iters = i
        prev_time = sys.clock()
        for j = 1,batches_per_epoch do
            -- local start = os.clock()
    	    local minibatch_targets,minibatch_inputs = unpack(self.batch_iterator())
            -- local loading_time = os.clock() - start
            -- local start = os.clock()
            num_processed = num_processed + self.general_config.batch_size
            self:train_batch(minibatch_inputs,minibatch_targets) 
            -- local training_time = os.clock() - start
            -- print(loading_time.." "..training_time)
        end
        local avg_error = self.total_error/batches_per_epoch
        local curr_time = sys.clock()
        local elapsed_time = curr_time - prev_time
        local rate = num_processed/elapsed_time
        num_processed = 0
        print(string.format('\nIter: %d\navg loss in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avg_error, elapsed_time,elapsed_time/batches_per_epoch))
        print(string.format('examples/sec = %f',rate))
        print('parameters norm: '..self.parameters:norm())

        self.data_for_callbacks.epoch = i
        for _,c in pairs(self.callbacks) do
            c:run(self.data_for_callbacks)
       end

    end
end

function  Train:pre_batch()
    --optional abstract method to be implemented by children
end

function Train:train_batch(inputs, targets)
    assert(inputs)
    assert(targets)

    self:pre_batch()

    local function f_eval(x)
        assert(x == self.parameters)
        self.grad_parameters:zero()
        local err = self.loss_wrapper:accumulate_gradient(inputs,targets)
        if(self.clamp_gradient) then
            local norm = self.grad_parameters:norm()
            if(norm > self.gradient_clip) then
                self.grad_parameters:mul(self.gradient_clip/norm)
            end
        end

        self.total_error = self.total_error + err
        if(self.general_config.assert_nan) then assert(self.grad_parameters:eq(self.grad_parameters):all(),'NANs in grad parameters') end

        return err, self.grad_parameters
    end
    self.optimization_config.opt_method(f_eval, self.parameters, self.optimization_config.opt_config, self.optimization_config.opt_state)
    if(self.general_config.post_process_parameter_update) then
        self.general_config.post_process_parameter_update()
    end
    if(self.general_config.assert_nan) then assert(self.parameters:eq(self.parameters):all(),'NANs in parameters after gradient step') end
end


