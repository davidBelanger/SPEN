
require 'Imports'

package.path = package.path .. ';model/?.lua'
require 'ChainCRF'
package.path = package.path .. ';infer1d/?.lua'
require 'SumProductInference'
require 'ExactInference'
require 'Inference1DUtil'

local config = {}
config.batch_size = 5
config.length = 4
config.domain_size = 3
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape
config.feature_width = 3
config.feature_hid_size = 5
config.local_potentials_scale = 1.0
config.pairwise_potentials_scale = 1.0


local crf_model = ChainCRF(config)

local x = torch.randn(config.batch_size,config.length,config.feature_size)
local log_edge_potentials_value = crf_model.log_edge_potentials_network:forward(x)



print('SumProduct')
local sum_product = SumProductInference(y_shape)
local predicted_edge_marginals, logZ = sum_product:infer_values(log_edge_potentials_value)
Inference1DUtil:assert_calibrated_edge_marginals(predicted_edge_marginals, y_shape) 
local predicted_node_marginals = Inference1DUtil:edge_marginals_to_node_marginals(predicted_edge_marginals, y_shape)
print(predicted_node_marginals[1])

assert(config.length == 4)
print('EXACT')
local exact_log_z, exact_marginals = ExactInference(y_shape):infer_values(log_edge_potentials_value)
local exact_node_marginals = Inference1DUtil:edge_marginals_to_node_marginals(exact_marginals, y_shape)
assert((logZ - exact_log_z):norm() < 0.00001)
assert((exact_node_marginals - predicted_node_marginals):norm() < 0.00001)
assert((exact_marginals - predicted_edge_marginals):norm() < 0.0001)
assert((logZ - exact_log_z):norm()< 0.0001)
print(exact_node_marginals[1])


