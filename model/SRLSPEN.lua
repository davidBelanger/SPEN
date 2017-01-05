require 'SPEN'
require 'Inference1DUtil'
local SRLSPEN, parent = torch.class('SRLSPEN','SPEN')

function SRLSPEN:__init(config, params)
	self.config = config
	self.config.node_feature_dim = 300 --todo: surface this option
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
		local_potentials = nn.SpatialConvolution(self.config.feature_dim,self.domain_size,1,1)(edge_features)
	else
		local flattened = nn.View(self.batch_size,self.num_rows*self.num_cols,self.config.feature_dim)(edge_features)
		local_potentials = 	self.nn.TemporalConvolution(self.config.feature_dim,self.domain_size,1,1)(flattened)
		local_potentials = nn.View(self.batch_size,self.num_rows,self.num_cols,self.domain_size)(local_potentials)
	end
	return nn.gModule({conditioning_values},{local_potentials})
end

function SRLSPEN:in_degree_energy(non_null_mass)
	assert(false,'change this to have different weights per predicate label/feature')
	local function in_degree_score(dim)
		local in_degree = nn.Sum(dim)(non_null_mass) --b x n 
		local score = nn.Mul(0,true)(nn.SoftPlus()(nn.Add(0,true)(nn.Mul(0,true)(in_degree))))
		return nn.Mul()(nn.Mean(2)(score))
	end
	
	local col_in_degree_score = in_degree_score(2)
	local row_in_degree_score = in_degree_score(3)
	return nn.CAddTable()({col_in_degree_score,row_in_degree_score})
end


function SRLSPEN:sibling_energy(non_null_mass, predicate_features, argument_features)

	-- For each predicate, non-linearly pool the features of the arguments that are pointing to it. 
	-- Then, score these with the features of the predicate

	local feature_dim = self.config.node_feature_dim
	local y_expand = nn.Replicate(feature_dim,4)(non_null_mass)

	local arg_features_expand = nn.Replicate(self.num_rows,2)(argument_features)
	local scaled_features = nn.CMulTable()({arg_features_expand,y_expand})
	local attachment_features = nn.Mean(3)(scaled_features) --b x n x f

	local merged_features = nn.JoinTable(3)({attachment_features,predicate_features})

	local h =  nn.SoftPlus()(self.nn.TemporalConvolution(2*feature_dim,self.config.energy_hid_size,1)(merged_features))
	local predicate_scores = nn.Mean(2)(nn.TemporalConvolution(self.config.energy_hid_size,1,1)(h))

	return nn.View(self.batch_size)(predicate_scores)
end


function SRLSPEN:sibling_arc_label_energy(y_non_null, predicate_features)
	--y_non_null: b x p x a x l
	--get features for the bag of labels for every predicate
	local bag_of_labels = nn.Sum(3)(y_non_null) --b x p x l
	local h =  nn.SoftPlus()(self.nn.TemporalConvolution(self.domain_size - 1,self.config.energy_hid_size,1)(bag_of_labels))
	local joined_features = nn.JoinTable(3)({predicate_features,h})
	local h2 =  nn.SoftPlus()(self.nn.TemporalConvolution(self.config.node_feature_dim + self.config.energy_hid_size,self.config.energy_hid_size,1)(joined_features))
	local predicate_scores = nn.Mean(2)(nn.TemporalConvolution(self.config.energy_hid_size,1,1)(h2))

	return nn.View(self.batch_size)(predicate_scores)
end

--this applies a non-linear score to the edges that are used
function SRLSPEN:topical_energy(non_null_mass,edge_features, feature_dim)
	local y_expand = nn.Replicate(feature_dim,4)(non_null_mass)

	local scaled_features = nn.CMulTable()({edge_features,y_expand})
	local global_average_used_arc = nn.Mean(2)(nn.View(self.batch_size,self.num_rows*self.num_cols,feature_dim)(scaled_features))

	local h =  nn.SoftPlus()(self.nn.TemporalConvolution(feature_dim,self.config.energy_hid_size,1)(global_average_used_arc))
	local predicate_scores = nn.Linear(self.config.energy_hid_size,1)(h)
	return nn.View(self.batch_size)(predicate_scores)
end

function SRLSPEN:outer_product(u,v,size1,size2)
	local u1 = nn.Reshape(size1[1], size1[2], 1, false)(u)
	local v1 = nn.Reshape(size2[1], size2[2], 1, false)(v)
	local prod = nn.MM(false,true)({u1,v1})
	return prod
end

function SRLSPEN:outer_sum(u,v,size1,size2)
	local u1 = nn.Reshape(size1[1], 1, size1[2], false)(u)
	local u1_tile = nn.Replicate(size2[2],2)(u1)
	local v1 = nn.Reshape(size2[1], size2[2], 1, false)(v)
	local v1_tile = nn.Replicate(size1[2],3)(v1)

	local prod = nn.CAddTable()({u1_tile,v1_tile})
	return prod
end


function SRLSPEN:self_collision_energy(non_null_mass, a2a_collisions)
	local a1 = nn.View(self.batch_size,self.num_rows,self.num_cols,1)(non_null_mass)
	local a1_tile = nn.Replicate(self.num_cols,4)(a1)

	local a2 = nn.View(self.batch_size,self.num_rows,1,self.num_cols)(non_null_mass)
	local a2_tile = nn.Replicate(self.num_cols,3)(a2)
	local outer_sum = nn.CAddTable()({a1_tile,a2_tile})

	local predicted_collisions = nn.Sum(2)(nn.Threshold(1.0)(outer_sum))

	local collision_predictions = nn.CMulTable()({a2a_collisions,predicted_collisions}) 
	local score = nn.Mul()(nn.Mean(2)(nn.Reshape(-1)(collision_predictions)))
	score.data.module.weight:fill(1.0)
	return score
end

function SRLSPEN:label_dependent_edge_features(edge_features,y_non_null)
	--edge_features: b x np x na x d
	--y_non_null: b x np x na x l
	--returns: b x np x na x f

	local non_null_domain_size = self.domain_size - 1
	--map features: b x np x na x d --> b x np x na x lf --> b x np x na x f x l
	local flattened_features = nn.View(self.batch_size,self.num_rows*self.num_cols,self.config.feature_dim)(edge_features)
	local flattened_label_dependent_features = 	self.nn.TemporalConvolution(self.config.feature_dim,non_null_domain_size*self.config.energy_hid_size,1,1)(flattened_features)
	local label_dependent_features = nn.View(self.batch_size,self.num_rows,self.num_cols,self.config.energy_hid_size,non_null_domain_size)(flattened_label_dependent_features)

	--map y: b x np x na x l --> b x np x na x f x l (by tiling)
	local y_expand = nn.View(self.batch_size,self.num_rows,self.num_cols, 1, non_null_domain_size)(y_non_null)
	local tiled_labels = nn.Replicate(self.config.energy_hid_size,4)(y_expand)

	local mul = nn.CMulTable()({label_dependent_features,tiled_labels})
	return nn.Sum(5)(mul)	 --b x np x na x f
end

function SRLSPEN:global_energy_net()
	local y = nn.Identity()()
	local conditioning_values = nn.Identity()()
	local energy_terms = {}

	assert(self.config.null_arc_index == 1)
	local y_non_null = nn.Narrow(4,2,self.domain_size - 1)(y) --this assumes that the null arc is the first label
	local non_null_mass = nn.Sum(4)(y_non_null)
	
	local hard_constraint_weight = self.config.hard_constraint_weight --todo: surface --TODO: this might be way too big
	local other_term_weight=1
	
	local arc_reward = nn.Mul()(nn.Mean(2)(nn.Reshape(-1)(non_null_mass)))
	arc_reward = nn.MulConstant(-other_term_weight,true)(arc_reward)
	table.insert(energy_terms,arc_reward)

	do 
		local collision_indicators = nn.SelectTable(2)(conditioning_values)

		local a2a_collisions = nn.SelectTable(3)(collision_indicators)
		local a2a_collision_energy = self:self_collision_energy(non_null_mass, a2a_collisions)

		local collision_energy = nn.MulConstant(10.0*hard_constraint_weight,true)(a2a_collision_energy) --todo:surface
		table.insert(energy_terms, collision_energy)
	end

	if(true) then
		local features = nn.SelectTable(1)(conditioning_values)
		local predicate_features = nn.SelectTable(2)(features)	
		local argument_features = nn.SelectTable(3)(features)

		-- This builds a representation for all arguments coming into a predicate, and then scores their compatibility with the predicate
		local sibling_energy = self:sibling_energy(non_null_mass, predicate_features, argument_features, 3)
		sibling_energy = nn.MulConstant(other_term_weight,true)(sibling_energy)
		table.insert(energy_terms, sibling_energy)

		-- -- This scores which sorts of arcs should associate with types of predicates
		local sibling_energy2 = self:sibling_arc_label_energy(y_non_null, predicate_features)
		sibling_energy2 = nn.MulConstant(other_term_weight,true)(sibling_energy2)
		table.insert(energy_terms, sibling_energy2)
	end

	--This represents the output as a bag of label-dependent edge features, pools these, and scores this.
	-- TODO: right now it's very computationally expensive
	if(false) then
		local features = nn.SelectTable(1)(conditioning_values)
		local edge_features = nn.SelectTable(1)(features)	
		local label_dependent_edge_features = self:label_dependent_edge_features(edge_features,y_non_null)
		local topical_energy = self:topical_energy(non_null_mass,label_dependent_edge_features, self.config.energy_hid_size)
		topical_energy = nn.MulConstant(other_term_weight,true)(topical_energy)
		table.insert(energy_terms, topical_energy)
	end

	local energy = (#energy_terms == 1) and energy_terms[1] or nn.CAddTable()(energy_terms)
	return nn.gModule({y,conditioning_values},{energy})
end


function SRLSPEN:features_net()
	return nn.SelectTable(1)
end


