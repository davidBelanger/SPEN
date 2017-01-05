require 'Imports'
package.path = package.path .. ';model/?.lua'
require 'ChainCRF'

package.path = package.path .. ';problem/?.lua'
require 'ChainCRFSequenceTagging'

package.path = package.path .. ';batch/?.lua'
require 'BatcherFromFile'

local outpath = "./data/sequence/crf-data"
local outmodel = "./data/sequence/true-model.torch"
config.batch_size = 5
config.length = 10
config.domain_size = 5
config.feature_size = 6
local y_shape = {config.batch_size,config.length,config.domain_size}
config.y_shape = y_shape

config.feature_width = 3
config.feature_hid_size = 5
config.local_potentials_scale = 2.5
config.pairwise_potentials_scale = 3.0

local data_config = {
    num_test_batches = 250,
    num_train_batches = 2500,
}

local problem = ChainCRFSequenceTagging(config, data_config)
torch.save(outmodel,problem.crf_model.log_edge_potentials_network)

local test_batcher  = problem:get_test_batcher()
local train_batcher = problem:get_train_batcher()
local test_file = outpath..".test"
local train_file = outpath..".train"
IO:save_batcher(train_batcher, data_config.num_train_batches, train_file)
IO:save_batcher(test_batcher, data_config.num_test_batches, test_file)

local train_batcher2 = BatcherFromFile({train_file}, nil, config.batch_size, false)
local test_batcher2 = BatcherFromFile({test_file}, nil, config.batch_size, false)


local function assertEqual(b1,b2)
	assert(#b1 == #b2)
	for i = 1,#b1 do
		if(torch.isTensor(b1[i])) then
			assert(b1[i]:double():eq(b2[i]:double()):all())
		else
			assert(b1[i] == b2[i])
		end
	end
end

local function assertBatchersEqual(b1,b2)
	local iter = b1:get_iterator()
	local iter2 = b2:get_iterator()

	for i = 1,15 do
		assertEqual(iter(), iter2())
	end
end

local b = train_batcher:get_iterator()()

assertBatchersEqual(train_batcher, train_batcher2)
assertBatchersEqual(test_batcher,  test_batcher2)

