
require 'Imports'
package.path = package.path .. ';model/?.lua'
require 'ChainCRF'

package.path = package.path .. ';problem/?.lua'
require 'ChainCRFSequenceTagging'

local outpath = "data/crf/tmp"
config.batch_size = 5
config.length = 10
config.domain_size = 5
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape

config.architecture = {}
config.architecture.feature_width = 3
config.architecture.hid_feature_size = 5
config.architecture.local_potentials_scale = 2.5
config.architecture.pairwise_potentials_scale = 3.0

local data_config = {
	num_test_batches = 250,
    num_train_batches = 2500,
}


local problem = ChainCRFSequenceTagging(config, data_config)


local test_batcher  = problem:get_test_batcher()
local train_batcher = problem:get_train_batcher()

IO:save_batcher(train_batcher, data_config.num_train_batches, outpath..".train")
IO:save_batcher(test_batcher, data_config.num_test_batches, outpath..".test")