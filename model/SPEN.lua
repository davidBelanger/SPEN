local SPEN = torch.class('SPEN')

-------------------Fields that you will likely access from outside the SPEN class.-----------------------------
-- spen.initialization_network :
-- spen.energy_network :
-- spen.features_network :
-- spen.classifier_network :
-- spen.global_potentials_network :
---------------------------------------------------------------------------------------------------------------


function SPEN:__init(config, params)
	params = params or {}
	self.params = params
	self.y_shape = config.y_shape
	self.config = config

	if(params.use_cuda and params.cudnn) then
		self.nn = cudnn
	else
		self.nn = nn
	end

	local energy_network, local_potentials_net, global_potentials_net = self:energy_net()
	self._local_potentials_net = nn.TruncatedBackprop(local_potentials_net)
	local features_network = nn.TruncatedBackprop(self:features_net())

	--the unaries_network is the local classifier, but without the features network
	local unaries_network = nn.Sequential():add(self._local_potentials_net):add(nn.MulConstant(-1,true)):add(self:normalize_unary_prediction()) --we multiply by -1 because inference does energy *minimization*, but local classification will maximize the score for a prediction. 

	local classifier_network = nn.Sequential():add(features_network):add(unaries_network)

	local initialization_network
	if(params.init_at_local_prediction and params.init_at_local_prediction == 1) then
		initialization_network = unaries_network
	else
		initialization_network = self:uniform_initialization_net()
	end

	--these are the networks that you'll want to access from outside
	--note that we don't seek to expose the _local_potentials_net. This is because it is easy to confuse this with the classifier network, which differs by a -1, and maybe some postprocessing.
	self.initialization_network = initialization_network
	self.energy_network = energy_network
	self.features_network = features_network
	self.classifier_network = classifier_network
	self.global_potentials_network = global_potentials_net

	assert(not config.dropout or (config.dropout == 0),'this functionality is not currently supported')

end


-------------------Methods that you may call for SPEN instances from outside the SPEN class.-----------------------------

--Setting this to true will prevent backprop through the features network.
function SPEN:set_feature_backprop(value)
	self.features_network:setBackprop(value)
end

--Setting this to true will prevent backprop through the features local potentials. 
function SPEN:set_unary_backprop(value)
	self._local_potentials_net:setBackprop(value)
end

-------------------Methods that you will need to implement in classes that extend SPEN. These are only called internally by the SPEN constructor. ------------------------------

function SPEN:features_net()
	os.error("children of SPEN should implement this method")
end

function SPEN:unary_energy_net()
	os.error("children of SPEN should implement this method")
end


function SPEN:global_energy_net()
	os.error("children of SPEN should implement this method")
end



-------------------Methods that you may want to override in children of SPEN-----------------------------------------

-- This may be overriden by children of SPEN
--TODO: do we actually need this?
function SPEN:convert_y_for_local_potentials(y)
	return y
end


-- This is an un-learned network for producing y0, the starting position for gradient-based inference.
-- Children may want to override this. For example, this may not make sense for problems with continuous labels.
function SPEN:uniform_initialization_net()
	local init_y_value = torch.zeros(unpack(self.y_shape))
	local domain_size = self.y_shape[#self.y_shape]
	if(not self.config.logit_iterates) then
		init_y_value:fill(1.0/domain_size)
	end

	return nn.Constant(init_y_value)
end


-------------------Methods that you won't need to implement/override are below this line-------------------------------

-- This assembles a network that takes {y,x} and returns a 1D Tensor with the energy for each element of the minibatch.
function SPEN:full_energy_net()
	local x = nn.Identity()()
	local y = nn.Identity()()
	local conditioning_values = self.features_network(x)
	local energy = self.energy_network({y,conditioning_values})
	return nn.gModule({y,x}, {energy})
end

-- This assembles a network that takes {y,features} and returns a 1D Tensor with the energy for each element of the minibatch.
-- During gradient-based optimization, you'll want to use this, and compute the features up front once.
function SPEN:energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local y = nn.Identity()()
	
	local local_potentials_net = self:unary_energy_net()
	local global_potentials_net = self:global_energy_net()


	local local_potentials = nn.Sum(2)(nn.Reshape(-1)(nn.CMulTable()({self:convert_y_for_local_potentials(y),local_potentials_net(conditioning_values)})))

	local global_potentials = global_potentials_net({y,conditioning_values})
	if(self.config.global_term_weight and self.config.global_term_weight ~= 1.0) then
		assert(self.config.global_term_weight >= 0)
		global_potentials = nn.MulConstant(self.config.global_term_weight,true)(global_potentials)
	end


	local energy = nn.CAddTable()({local_potentials, global_potentials})

	local energy_net = nn.gModule({y,conditioning_values},{energy})
	return energy_net, local_potentials_net, global_potentials_net

end


--todo: get back this functionality
-- function SPEN:addDropout(n,inplace,sizes)
-- 	local ip
-- 	if(inplace == nil) then
-- 		ip = true
-- 	else
-- 		ip = inplace
-- 	end
--     if(self.useDropout) then 
--     	local drop = nn.EpochDropout(self.params.dropout,false,sizes,ip)
--     	table.insert(self.dropouts,drop)
--         n:add(drop)
--     end
-- end
