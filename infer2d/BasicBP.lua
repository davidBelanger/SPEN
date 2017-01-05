local BasicBP = torch.class('BasicBP')


function BasicBP:infer(log_node_potentials,log_edge_potentials)
	local y_shape = torch.toList(log_node_potentials:size())
	local domain_size = y_shape[4]
	local node_potentials = log_node_potentials:clone():exp()
	local edge_potentials = log_edge_potentials:clone():exp()

	local function initialize() return torch.ones(log_node_potentials:size()):fill(1/domain_size) end

	local messages = 
end