require 'SPEN'
require 'Inference1DUtil'
local BinaryGraphSPEN, parent = torch.class('BinaryGraphSPEN','SPEN')



function BinaryGraphSPEN:__init(config)
	parent.__init(self,config)
	self.config = config
	self.batch_size, self.num_nodes, _, self.domain_size = unpack(config.y_shape)
	self.feature_size = config.feature_size

	self.energy_network = self:energy_net()
	self.features_network = self:features_net()

	self.classifier_network = nn.Sequential():add(self.features_network):add(self.local_potentials_net):add(Inference1DUtil:softmax_4d(config.y_shape))
	if(config.initialize_uniformly) then
		self.initialization_network = self:uniform_initialization_net()
	else
		self.initialization_network = self.classifier_network
	end

end

function BinaryGraphSPEN:features_net()
	return nn.Identity()
end

function BinaryGraphSPEN:energy_net()
	assert(self.domain_size == 2)
	local conditioning_values = nn.Identity()() --b x n x f
	local architecture = self.config.architecture


	local child_embeddings  = nn.TemporalConvolution(self.feature_size,architecture.hid_feature_size,1,1)(conditioning_values) --b x n x h
	local parent_embeddings = nn.TemporalConvolution(self.feature_size,architecture.hid_feature_size,1,1)(conditioning_values) --b x n x h
	local pos_scores = nn.MM(false,true)({child_embeddings,parent_embeddings})
	pos_scores = nn.Reshape(self.batch_size,self.num_nodes,self.num_nodes,1)(pos_scores)
	local neg_scores = nn.MulConstant(-1)(pos_scores)
	local scores = nn.JoinTable(4)({neg_scores,pos_scores})

	self.local_potentials_net = nn.gModule({conditioning_values},{scores})

	local conditioning_values = nn.Identity()() --b x l x f

	local y = nn.Identity()()
	local scores = self.local_potentials_net(conditioning_values)
	local local_potentials = nn.Sum(2)(nn.Reshape(self.batch_size,self.num_nodes*self.num_nodes*self.domain_size,false)(nn.CMulTable()({y,scores})))
	local energy = local_potentials
	local energy_net = nn.gModule({y,conditioning_values},{energy})
	return energy_net
end
