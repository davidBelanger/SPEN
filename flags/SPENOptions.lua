local SPENOptions = torch.class('SPENOptions')

function SPENOptions:add_general_spen_options(cmd)
	cmd:option('-global_term_weight',1.0,'multiplier to put on the non-unary energy term')
end


