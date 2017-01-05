require 'SPEN'
local DepthSPEN, parent = torch.class('DepthSPEN','SPEN')

function DepthSPEN:__init(config, params)
	self.config = config
	self.batch_size, self.height, self.width = unpack(config.y_shape)
	parent.__init(self, config, params)
end
 
function DepthSPEN:normalize_unary_prediction()
	return nn.Identity()
end
function DepthSPEN:features_net()
	-- local x = nn.Identity()() --b x l x f
	-- local x2 = nn.View(self.batch_size,1,self.height,self.width)(x)
	-- local ker_size = 5
	-- local pad_size = 0.5*(ker_size - 1)
	-- local features = self.nn.SpatialConvolution(1,self.config.feature_hid_size,ker_size,ker_size,1,1,pad_size,pad_size)(x2)
	-- features = nn.SoftPlus()(features)
	-- features = self.nn.SpatialConvolution(self.config.feature_hid_size,self.config.feature_hid_size,ker_size,ker_size,1,1,pad_size,pad_size)(x2)

	-- local all_features = {features,x}
	-- local features_net = nn.gModule({x},{features})
	-- return features_net
	
	return nn.Identity()
end

--unlike the global_energy_net, this doesn't return an actual energy value, it returns the sufficient satistics of a local model that defines an energy.
function DepthSPEN:unary_energy_net()
	-- local conditioning_values = nn.Identity()() --b x l x h
	-- local per_label_mean = nn.SpatialConvolution(self.config.feature_hid_size,1,1,1)(conditioning_values)
	-- per_label_mean = nn.View(self.batch_size,self.height,self.width)(per_label_mean)
	-- return nn.gModule({conditioning_values},{per_label_mean})
	
	return nn.Identity()
end

function DepthSPEN:global_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local y0 = nn.Identity()()

	local y = nn.SelectTable(1)({y0,conditioning_values}) --this is to insert conditioning_values into the graph
	y = nn.View(self.batch_size,1,self.height,self.width)(y)
	local ker_size = 5
	local pad_size = 0.5*(ker_size - 1)
	local features = self.nn.SpatialConvolution(1,self.config.energy_hid_size,ker_size,ker_size,1,1,pad_size,pad_size)(y)
	features = nn.SoftPlus()(features)
	features = self.nn.SpatialConvolution(self.config.energy_hid_size,self.config.energy_hid_size,ker_size,ker_size,1,1,pad_size,pad_size)(features)
	features = nn.SoftPlus()(features)

	local scores = self.nn.SpatialConvolution(self.config.energy_hid_size,1,1,1)(features)
	local potentials = nn.Sum(2)(nn.View(self.batch_size,self.height*self.width)(scores))
	return nn.gModule({y0,conditioning_values},{potentials})
end
