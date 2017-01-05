local BatcherFromFactory = torch.class('BatcherFromFactory')

function BatcherFromFactory:__init(factory,preprocess,num_instances, use_cuda)
	if(num_instances ~= nil) then
		self.instances = {}
		for t = 1,num_instances do
			local data = factory()
			local processed_data = preprocess and preprocess(data) or data
			table.insert(self.instances, processed_data)
		end
	end
	self.factory = factory
	self.num_instances = num_instances
	self.preprocess = preprocess
end 


function BatcherFromFactory:get_iterator(ongoing, shuffle)
	if(self.num_instances == nil) then
		return function () 
			local data = self.factory()
			local processed_data = self.preprocess and self.preprocess(data) or data
			return {processed_data[1],processed_data[2],processed_data[1]:size(1)}
		end
	else
		local idx = 0
		return function()
			idx = idx + 1
			if(ongoing and idx > self.num_instances) then
				idx = 1
				if(shuffle) then
					assert(false, 'not implemented')
				end
			end
			local processed_data = self.instances[idx]
			if(not processed_data) then  return nil end
			return {processed_data[1],processed_data[2],processed_data[1]:size(1)}
		end
	end
end

function BatcherFromFactory:get_ongoing_iterator(shuffle)
	return self:get_iterator(true)
end