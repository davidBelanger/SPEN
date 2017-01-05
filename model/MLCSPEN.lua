require 'SPEN'
local MLCSPEN, parent = torch.class('MLCSPEN','SPEN')


--todo: need to refactor the features net stuff a bit so that it can share more stuff with the pretraining net


function MLCSPEN:__init(config, params)
	self.config = config
	self.batch_size, self.num_labels, domain_size = unpack(config.y_shape)
	assert(domain_size == 2)

	parent.__init(self, config, params)

end

function MLCSPEN:normalize_unary_prediction()
	return nn.Sequential():add(nn.Sigmoid()):add(self:expand_to_probs())
end
 
function MLCSPEN:expand_to_probs()
	local input = nn.Identity()()
	local p_neg = nn.AddConstant(1,true)(nn.MulConstant(-1)(input))
	p_pos = nn.Reshape(self.batch_size,self.num_labels,1)(input)
	p_neg = nn.Reshape(self.batch_size,self.num_labels,1)(p_neg)

	local probs = nn.JoinTable(3)({p_neg,p_pos})
	return nn.gModule({input},{probs})
end

function MLCSPEN:features_net()
	local x = nn.Identity()() --b x l x f

	local h = x
	local cur_size = self.config.input_size
	for i = 1,self.config.feature_depth do
		h = nn.Linear(cur_size,self.config.feature_hid_size)(h)
		h = nn[self.config.energy_nonlinearity]()(h)
		cur_size = self.config.feature_hid_size
	end

	 local features_net = nn.gModule({x},{h})
	 return features_net
end


function MLCSPEN:unary_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local pos_scores = nn.Linear(self.config.feature_hid_size,self.num_labels)(conditioning_values) --this is the energy for the labels being on. 
	return nn.gModule({conditioning_values},{pos_scores})
end

function MLCSPEN:convert_y_for_local_potentials(y)
	return nn.Select(3,2)(y)
end

--this is assumed to take a batchsize x num_labels tensor for y
function MLCSPEN:global_energy_net()
	--todo: dropout
	local y = nn.Identity()()
	local conditioning_values = nn.Identity()()
	local y_pos = nn.Select(3,2)(y)


	local input_to_joint_energy, input_dim_to_joint_energy
	if(self.config.conditional_label_energy) then
		input_to_joint_energy = nn.JoinTable(2)({y_pos,conditioning_values})
		input_dim_to_joint_energy = self.num_labels + self.config.feature_hid_size
	else
		input_to_joint_energy = y_pos
		input_dim_to_joint_energy = self.num_labels
	end
	
	local cur_size = input_dim_to_joint_energy
	local h = input_to_joint_energy
	for i = 1,self.config.energy_depth do
		h = nn.Linear(cur_size,self.config.energy_hid_size)(h)
		h = nn[self.config.energy_nonlinearity]()(h)
		cur_size = self.config.energy_hid_size
	end

	local global_potentials = nn.Select(2,1)(nn.Linear(cur_size,1)(h))

	return nn.gModule({y,conditioning_values},{global_potentials})
end
