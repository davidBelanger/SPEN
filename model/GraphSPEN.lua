require 'SPEN'
require 'Inference1DUtil'
local GraphSPEN, parent = torch.class('GraphSPEN','SPEN')



function GraphSPEN:__init(config)
	parent.__init(self,config)
	self.config = config
	self.batch_size, self.num_cols, self.num_rows, self.domain_size = unpack(config.y_shape)
	
	parent.__init(self, config, params)


end

function GraphSPEN:normalize_unary_prediction()
	return Inference1DUtil:softmax_4d(self.config.y_shape)
end
end

function GraphSPEN:unary_prediction_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local flattened = nn.Reshape(self.batch_size,self.num_cols*self.num_rows,self.config.hid_feature_size,false)(conditioning_values)
	local local_potentials = nn.TemporalConvolution(self.config.hid_feature_size,self.config.domain_size,1)(flattened)
	local_potentials = nn.Reshape(self.batch_size,self.num_cols,self.num_rows,self.config.hid_feature_size)(local_potentials)
end


function GraphSPEN:global_energy_net()
	local y = nn.Identity()()
	local conditioning_values = nn.Identity()()

	local y_non_null = nn.Narrow(4,1,self.domain_size - 1)(y) --this assumes that the null arc is the final label
	local non_null_mass = nn.Sum(4)(y_non_null)
	
	local function in_degree_score(dim)
		local in_degree = nn.Sum(dim)(y_non_null) --b x n 
		local score = nn.SoftPlus()(nn.Add(0,true)(nn.Mul(0,true)(in_degree)))
		return nn.Sum(2)(score)
	end
	
	local col_in_degree_score = in_degree_score(2)
	local row_in_degree_score = in_degree_score(3)

	local energy = nn.CAddTable()({col_in_degree_score,row_in_degree_score})
	return nn.gModule({y,conditioning_values},{energy})
end


function GraphSPEN:features_net()
--	local x = nn.Identity()() --b x n x f
	return nn.Identity()
end


