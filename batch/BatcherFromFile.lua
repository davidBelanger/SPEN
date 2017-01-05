local BatcherFromFile = torch.class('BatcherFromFile')
function BatcherFromFile:__init(file_list,preprocess,batch_size,use_cuda)
	self.file_list = file_list
	self.preprocess = preprocess
	self.use_cuda = use_cuda
	self.batch_size = batch_size
end

--todo: we should really be using torchnet...
function BatcherFromFile:get_iterator()
	local shuffle = false
	if(self.one_pass_batcher) then
		self.one_pass_batcher:reset()
	end
	self.one_pass_batcher = self.one_pass_batcher or OnePassMiniBatcherFromFileList(self.file_list,self.batch_size,self.use_cuda,self.preprocess,shuffle)
	return function()
		local labels, data, num_actual_data = self.one_pass_batcher:getBatch()
		if(labels == nil) then  return nil end

		return {labels, data, num_actual_data}
	end
end

function BatcherFromFile:get_ongoing_iterator(shuffle)
	local batcher = MinibatcherFromFileList(self.file_list,self.batch_size,self.use_cuda,self.preprocess,shuffle)
	return function()
		local labels, data, num_actual_data = batcher:getBatch()
		if(labels == nil) then  return nil end
		return {labels, data, num_actual_data}
	end
end

