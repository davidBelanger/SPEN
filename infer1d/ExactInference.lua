require 'Inference'
local ExactInference, parent = torch.class('ExactInference')


function ExactInference:__init(y_shape)
	self.y_shape = y_shape
	self.batch_size, self.length, self.domain_size = unpack(y_shape)
	assert(self.length == 4)

end
function ExactInference:infer_values(log_edge_potentials)
	local edge_potentials = log_edge_potentials:clone():exp()
	local Z_Sums = torch.Tensor(self.batch_size)
	local Z_Marg_Sums = torch.zeros(torch.LongStorage({self.batch_size,self.length-1,self.domain_size,self.domain_size}))
	assert(edge_potentials:size(2) == 3)
	for b = 1,self.batch_size do
		for i1 = 1,self.domain_size do
			for i2 = 1,self.domain_size do
				for i3 = 1,self.domain_size do
					for i4 = 1,self.domain_size do
						score = edge_potentials[b][1][i1][i2] * edge_potentials[b][2][i2][i3] * edge_potentials[b][3][i3][i4]
						Z_Marg_Sums[b][1][i1][i2] = Z_Marg_Sums[b][1][i1][i2] + score 
						Z_Marg_Sums[b][2][i2][i3] = Z_Marg_Sums[b][2][i2][i3] + score
						Z_Marg_Sums[b][3][i3][i4] = Z_Marg_Sums[b][3][i3][i4] + score
						Z_Sums[b] = Z_Sums[b] + score
					end
				end
			end
		end
	end

	local exact_edge_marginals = Z_Marg_Sums:cdiv(Z_Sums:view(self.batch_size,1,1,1):expandAs(Z_Marg_Sums))
	local exact_log_z = nn.Log():forward(Z_Sums)
	return exact_log_z, exact_edge_marginals
end

