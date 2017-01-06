require 'SPEN'
local ChainSPEN, parent = torch.class('ChainSPEN','SPEN')

function ChainSPEN:__init(config, params)
	self.config = config
	self.batch_size, self.length, self.domain_size = unpack(config.y_shape)
	assert(self.length > 2)
	self.feature_size = config.feature_size

	parent.__init(self, config, params)
end
 

function ChainSPEN:normalize_unary_prediction()
	return Inference1DUtil:softmax_3d(self.config.y_shape)
end

function ChainSPEN:features_net()
	local x = nn.Identity()() --b x l x f

	local ker_width = self.config.feature_width
	local pad_size = 0.5*(ker_width - 1)
	local left_pad = nn.Constant(torch.zeros(self.batch_size,pad_size,self.feature_size))(x)
	local right_pad = nn.Constant(torch.zeros(self.batch_size,pad_size,self.feature_size))(x)
	local x_pad = nn.JoinTable(2,3)({left_pad,x,right_pad})
	local f1 = self.nn.TemporalConvolution(self.feature_size,self.config.feature_hid_size,3,1)(x_pad)
	local features = nn.ReLU()(f1)

	 local features_net = nn.gModule({x},{features})
	 return features_net
end

function ChainSPEN:unary_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local per_label_score = self.nn.TemporalConvolution(self.config.feature_hid_size,self.domain_size,1,1)(conditioning_values)
	return nn.gModule({conditioning_values},{per_label_score})
end

function ChainSPEN:global_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local y = nn.Identity()()

	local input_to_joint_energy, input_dim_to_joint_energy
	if(self.config.data_independent_joint_energy) then
		assert(false,'this should fail, since conditioning_values is not used')
		input_to_joint_energy = y
		input_dim_to_joint_energy = self.domain_size
	else
		input_to_joint_energy = nn.JoinTable(3)({y,conditioning_values})
		input_dim_to_joint_energy = self.domain_size + self.config.feature_hid_size
	end
	
	local energy_hid_size = self.config.energy_hid_size
	local pairwise_potentials = nn.SoftPlus()(self.nn.TemporalConvolution(input_dim_to_joint_energy,energy_hid_size,2,1)(input_to_joint_energy))
	pairwise_potentials = nn.SoftPlus()(self.nn.TemporalConvolution(energy_hid_size,energy_hid_size,1,1)(pairwise_potentials))
	pairwise_potentials = nn.Sum(2)(nn.Reshape(-1)(self.nn.TemporalConvolution(energy_hid_size,1,1,1)(pairwise_potentials)))


	return nn.gModule({y,conditioning_values},{pairwise_potentials})
end
