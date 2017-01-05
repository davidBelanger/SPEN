require 'SPEN'
require 'Inference1DUtil'
local SRLSPEN, parent = torch.class('SRLSPEN','SPEN')



function SRLSPEN:__init(config, params)
	self.config = config
	self.config.node_feature_dim = (self.config.feature_dim-1) * 0.5 --todo: surface this option
	self.batch_size, self.num_rows, self.num_cols, self.domain_size = unpack(config.y_shape)
	parent.__init(self, config, params)
end

function SRLSPEN:normalize_unary_prediction()
	return Inference1DUtil:softmax_4d(self.config.y_shape)
end


function SRLSPEN:unary_energy_net()
	local conditioning_values = nn.Identity()() --b x l x h
	local edge_features = nn.SelectTable(1)(nn.SelectTable(1)(conditioning_values))
	local local_potentials
	if(false and self.params.use_cuda) then
		edge_features = nn.Transpose({3,2})(nn.Transpose({4,3})(edge_features))
		--todo: very confused by why this doesn't work
		local_potentials = nn.SpatialConvolution(self.config.feature_dim,self.config.domain_size,1,1)(edge_features)
	else
		local flattened = nn.View(self.batch_size,self.num_rows*self.num_cols,self.config.feature_dim)(edge_features)
		local_potentials = 	self.nn.TemporalConvolution(self.config.feature_dim,self.config.domain_size,1,1)(flattened)
		local_potentials = nn.View(self.batch_size,self.num_rows,self.num_cols,self.config.domain_size)(local_potentials)
	end
	return nn.gModule({conditioning_values},{local_potentials})
end

function SRLSPEN:in_degree_energy(non_null_mass)

	local function in_degree_score(dim)
		local in_degree = nn.Sum(dim)(non_null_mass) --b x n 
		local score = nn.Mul(0,true)(nn.SoftPlus()(nn.Add(0,true)(nn.Mul(0,true)(in_degree))))
		return nn.Mul()(nn.Mean(2)(score))
	end
	
	local col_in_degree_score = in_degree_score(2)
	local row_in_degree_score = in_degree_score(3)
	return nn.CAddTable()({col_in_degree_score,row_in_degree_score})
end


--TODO: check this
function SRLSPEN:sibling_energy(non_null_mass, node_features, edge_features, summation_index)
	local feature_dim = self.config.node_feature_dim

	local y_expand = nn.Replicate(feature_dim,4)(non_null_mass)
	local node_features_expand = nn.Replicate(self.config.y_shape[summation_index],summation_index)(node_features)

	local scaled_features = nn.CMulTable()({node_features_expand,y_expand})
	local attachment_features = nn.Mean(summation_index)(scaled_features) --b x n x f

	local merged_features = nn.JoinTable(3)({attachment_features,node_features})

	local h =  nn.SoftPlus()(self.nn.TemporalConvolution(2*feature_dim,self.config.energy_hid_size,1)(merged_features))
	local predicate_scores = nn.Mean(2)(nn.TemporalConvolution(self.config.energy_hid_size,1,1)(h))

	return nn.View(self.batch_size)(predicate_scores)
end


--this gets a feature vector for all siblings of a predicate (row) and then scores them
-- function SRLSPEN:sibling_arc_energy(y_non_null,edge_features)
-- 	local feature_dim = self.config.feature_dim

-- 	local y_expand = nn.Replicate(feature_dim,4)(y_non_null)
-- 	local scaled_features = nn.CMulTable()({edge_features,y_expand})
-- 	local predicate_features = nn.Mean(3)(scaled_features)

-- 	local h =  nn.SoftPlus()(self.nn.TemporalConvolution(feature_dim,self.config.energy_hid_size,1)(predicate_features))
-- 	local predicate_scores = nn.Mean(2)(nn.TemporalConvolution(self.config.energy_hid_size,1,1)(h))

-- 	return nn.View(self.batch_size)(predicate_scores)
-- end

--this applies a non-linear score to the edges that are used
function SRLSPEN:topical_energy(non_null_mass,edge_features)
	local feature_dim = self.config.feature_dim

	local y_expand = nn.Replicate(feature_dim,4)(non_null_mass)

	local scaled_features = nn.CMulTable()({edge_features,y_expand})
	local global_average_used_arc = nn.Mean(2)(nn.View(self.batch_size,self.num_rows*self.num_cols,self.config.feature_dim)(scaled_features))

	local h =  nn.SoftPlus()(self.nn.TemporalConvolution(feature_dim,self.config.energy_hid_size,1)(global_average_used_arc))
	local predicate_scores = nn.Linear(self.config.energy_hid_size,1)(h)
	return nn.View(self.batch_size)(predicate_scores)
end

function SRLSPEN:label_compatibility_energy(y_non_null, summation_index)
	--y_non_null: b x n x m x l
	local incoming_arcs = nn.Mean(summation_index)(y_non_null)
	local feats = self.nn.TemporalConvolution(self.domain_size - 1, self.config.energy_hid_size,1)(incoming_arcs)
	local h = nn.SoftPlus()(feats)
	local scores = nn.Mean(2)(self.nn.TemporalConvolution(self.config.energy_hid_size,1,1)(h))
	scores = nn.View(self.batch_size)(scores)
	return scores
end

function SRLSPEN:outer_product(u,v,size1,size2)
	local u1 = nn.Reshape(size1[1], size1[2], 1, false)(u)
	local v1 = nn.Reshape(size2[1], size2[2], 1, false)(v)
	local prod = nn.MM(false,true)({u1,v1})
	return prod
end


function SRLSPEN:collision_energy(non_null_mass, p2a_collisions)
	--local thresh = 0.15
	--local arc_usage = nn.Threshold(thresh,0,false)(non_null_mass)
	local arc_usage = non_null_mass
	local collision_predictions = nn.CMulTable()({p2a_collisions,arc_usage}) 
	local score = nn.Mul()(nn.Mean(2)(nn.Reshape(-1)(collision_predictions)))
	score.data.module.weight:fill(1.0)

	return score
end

function SRLSPEN:self_collision_energy(non_null_mass, a2a_collisions, dimension)
	--local thresh = 0.15
	local summation_dimension = dimension == 2 and 3 or dimension == 3 and 2

	local arc_usage = nn.Sum(summation_dimension)(non_null_mass) --sum over the predicates
	--arc_usage = nn.Threshold(thresh,0,false)(arc_usage)
	local size = {self.batch_size,self.y_shape[dimension]}
	--todo: this should be re-implemented so that we avoid instantiating the outer product
	local predicted_collisions = self:outer_product(arc_usage,arc_usage,size,size)

	local collision_predictions = nn.CMulTable()({a2a_collisions,predicted_collisions}) 
	local score = nn.Mul()(nn.Mean(2)(nn.Reshape(-1)(collision_predictions)))
	score.data.module.weight:fill(1.0)

	return score
end

-- function SRLSPEN:a2a_collision_energy(non_null_mass, a2a_collisions)
-- 	local thresh = 0.15
-- 	local arc_usage = nn.Sum(2)(non_null_mass) --sum over the predicates
-- 	arc_usage = nn.Threshold(thresh,0,false)(arc_usage)
-- 	local size = {self.batch_size,self.num_cols}
-- 	--todo: this should be re-implemented so that we avoid instantiating the outer product
-- 	local predicted_collisions = self:outer_product(arc_usage,arc_usage,size,size)

-- 	local collision_predictions = nn.CMulTable()({a2a_collisions,predicted_collisions}) 
-- 	local score = nn.Mul()(nn.Mean(2)(nn.Reshape(-1)(collision_predictions)))
-- 	return score
-- end

function SRLSPEN:global_energy_net()
	local y = nn.Identity()()
	local conditioning_values = nn.Identity()()
	local energy_terms = {}

	assert(self.config.null_arc_index == 1)
	local y_non_null = nn.Narrow(4,2,self.domain_size - 1)(y) --this assumes that the null arc is the first label
	local non_null_mass = nn.Sum(4)(y_non_null)
	

	do 
		local collision_indicators = nn.SelectTable(2)(conditioning_values)

		local p2p_collisions = nn.SelectTable(1)(collision_indicators)
		local p2p_collision_energy = self:self_collision_energy(non_null_mass, p2p_collisions, 2)

		local p2a_collisions = nn.SelectTable(2)(collision_indicators)
		local p2a_collision_energy = self:collision_energy(non_null_mass, p2a_collisions)

		local a2a_collisions = nn.SelectTable(3)(collision_indicators)
		local a2a_collision_energy = self:self_collision_energy(non_null_mass, a2a_collisions, 3)

		local collision_energy = nn.CAddTable()({a2a_collision_energy,p2a_collision_energy,p2p_collision_energy})
		table.insert(energy_terms, collision_energy)
	end

	-- local in_degree_energy = self:in_degree_energy(non_null_mass)
	-- table.insert(energy_terms,in_degree_energy)
	
	-- local features = nn.SelectTable(1)(conditioning_values)
	-- local edge_features = nn.SelectTable(1)(features)
	-- local predicate_features = nn.SelectTable(2)(features)	
	-- local sibling_energy = self:sibling_energy(non_null_mass,predicate_features, edge_features, 3)
	-- table.insert(energy_terms, sibling_energy)

	-- -- local sibling_energy = self:sibling_arc_energy(y_non_null,edge_features)
	-- -- table.insert(energy_terms, sibling_energy)

	
	-- local topical_energy = self:topical_energy(non_null_mass,edge_features)
	-- table.insert(energy_terms, topical_energy)

	-- local predicate_arc_compatibility_energy = self:label_compatibility_energy(y_non_null,3)
	-- table.insert(energy_terms, predicate_arc_compatibility_energy)


	local energy = (#energy_terms == 1) and energy_terms[1] or nn.CAddTable()(energy_terms)
	return nn.gModule({y,conditioning_values},{energy})
end


function SRLSPEN:features_net()
	return nn.SelectTable(1)
end


