local GradientBasedInferenceConfig = torch.class('GradientBasedInferenceConfig')

function GradientBasedInferenceConfig:get_gd_inference_config(params)
	local optimization_config =   {
	    num_iterations =  params.max_inference_iters,
	    learning_rate_scale = params.inference_learning_rate,
	    momentum_gamma = params.inference_momentum,
	    line_search = params.line_search == 1,
	    init_line_search_step = init_line_search_step,	
	    rtol = params.inference_rtol,
	    learning_rate_decay = inference_learning_rate_decay,
	    return_optimization_diagnostics = true
	}

	local gd_inference_config = {
	    return_objective_value = false,
	    entropy_weight = params.entropy_weight,
	    logit_iterates = params.unconstrained_iterates == 1 and not (params.continuous_outputs == 1),
	    mirror_descent = params.mirror_descent == 1,
	    return_optimization_diagnostics = false,
	    optimization_config = optimization_config,
	    return_all_iterates = params.return_all_iterates,
	    continuous_outputs = params.continuous_outputs == 1
	}

	return gd_inference_config

end
