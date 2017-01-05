require 'nn'
require 'nngraph'
require 'Imports'
package.path = package.path .. ';model/?.lua'
require 'ChainCRF'
package.path = package.path .. ';infer1d/?.lua'
require 'Inference1DUtil'

local config = {}
config.batch_size = 5
config.length = 4
config.domain_size = 3
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape

config.= {}
config.feature_width = 3
config.feature_hid_size = 5
config.local_potentials_scale = 1.0
config.pairwise_potentials_scale = 1.0


local crf_model = ChainCRF(config)
local x = torch.randn(config.batch_size,config.length,config.feature_size)


local accumulator = torch.zeros(config.batch_size,config.length-1,config.domain_size,config.domain_size)

local log_edge_potentials_value = crf_model.log_edge_potentials_network:forward(x)
local sum_product = SumProductInference(y_shape)
local predicted_edge_marginals, logZ = sum_product:infer_values(log_edge_potentials_value)

nSamples = 3000
for i = 1,nSamples do
	local sample = crf_model:sample_from_joint(x)
	local expanded_sample = Inference1DUtil:make_outer_product_marginals_from_onehots(sample,config.y_shape)
	accumulator:add(expanded_sample)
	if(i % 250 == 0) then
		local avg = accumulator:clone():div(i)
		print((predicted_edge_marginals - avg):norm()/predicted_edge_marginals:norm())
	end
end

local avg = accumulator:clone():div(nSamples)
assert((predicted_edge_marginals - avg):norm()/predicted_edge_marginals:norm() < 0.05)
