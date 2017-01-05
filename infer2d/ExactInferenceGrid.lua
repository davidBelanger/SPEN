-- local ExactInferenceGrid =  torch.class('ExactInferenceGrid')

-- function ExactInferenceGrid:infer(local_potentials,pairwise_potentials)
-- 	local batch_size, height, width, domain_size = unpack(torch.totable(local_potentials:size()))

-- 	assert(height == 3 and width == 3)

-- 	local accum = torch.zeros(batch_size,height,width,domain_size)
-- 	for b = 1,batch_size do
-- 		for i1 = 1,domain_size do
-- 			for i2 = 1,domain_size do
-- 				for i3 = 1,domain_size do
-- 					for i4 = 1,domain_size do
-- 						for i5 = 1,domain_size do
-- 							for i6 = 1,domain_size do
-- 								for i7 = 1,domain_size do
-- 									for i8 = 1,domain_size do
-- 										for i9 = 1,domain_size do
-- 											energy = 0
-- 											accum[b][h][w]
-- 										end
-- 									end
-- 								end
-- 							end
-- 						end
-- 					end
-- 				end
-- 			end
-- 		end
-- 	end

-- -- 	return accum
-- -- end