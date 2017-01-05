local GeneralOptions = torch.class('GeneralOptions')


function GeneralOptions:get_flags()
	local cmd = torch.CmdLine()

	--NOTE: Additional Options are in flags/TrainingOptions.lua and flags/MultiLabelClassificationOptions.lua

	--General Options
	cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
	cmd:option('-cudnn',1,'whether to use cudnn, where appropriate')
	cmd:option('-batch_size',32,'minibatch size')
	cmd:option('-profile',0,"whether to do profiling")
	cmd:option('-shuffle',1,'whether to shuffle the data after you take a pass through it') --todo: needs reboot
	cmd:option('-problem','SequenceTagging',"what kind of problem (SequenceTagging,MultiLabelClassification,Denoise)")
	cmd:option('-training_configs',"","list of serialized training config files")
	cmd:option('-problem_config',"","serialized problem config file")

	--Test-Time Usage
	cmd:option('-evaluate_classifier_only',0,"whether to only evaluate the classifier model and then exit") 
	cmd:option('-evaluate_spen_only',0,"whether to only evaluate the full spen model and then exit") 

	--Data Options
	cmd:option('-train_list','','list of torch format train files')
	cmd:option('-test_list','','list of torch format dev/test files')
	cmd:option('-continuous_outputs',0,'whether the outputs are continuous or discrete')

	--Pretrained Parameters Options
	cmd:option('-init_classifier',"","where to load pretrained feature network from")
	cmd:option('-init_full_net',"","where to load pretrained parameters for full network from") 
	--cmd:option('-init_opt_state',"","where to load optimization state from") 

	cmd:option('-icnn',0,"whether to constrain the parameters in the energy network to be positive, which results in convexity wrt y.")


	cmd:option('-training_method',"E2E","General Training Method")

	--Inference Options
	cmd:option('-line_search',1,"whether to do line search")
	cmd:option('-init_line_search_step',1.0,"initial step size for backtracking line search")
	cmd:option('-inference_rtol',0.00001,"initial step size for backtracking line search")
	cmd:option('-penalize_all_iterates',0,'whether to apply the loss to iterates besides the final one. (If 0, then convergence regularization is not used)')
	cmd:option('-first_iter_to_apply_loss',10,"first iterate beyond which to penalize the loss")

	cmd:option('-convergence_regularization_weight',0.0,'weight to place on term the regularizes consecutive iterates to speed up convergence')
	cmd:option('-first_iter_to_penalize_convergence',10,"first iterate beyond which to penalize the convergence criterion")

	cmd:option('-inference_learning_rate',0.1,"learning rate for inference")
	cmd:option('-inference_learning_rate_power',1.0,"learning rate power for inference")

	cmd:option('-max_inference_iters',30,"how many inference iters to perform")
	cmd:option('-learn_inference_hyperparams',1,'whether to learn the inference hyperparams (eg. learning rates)')
	cmd:option('-inference_momentum',0,"") 
	cmd:option('-inference_learning_rate_decay',0,"not used if learn_inference_hyperparams is true") 
	cmd:option('-unconstrained_iterates',1,"whether to use logits as inference iterates") 
	cmd:option('-mirror_descent',1,"whether to use mirror descent for simplex-constrained problems") 
	cmd:option('-entropy_weight',1.0,"weight to place on entropy term") 
	cmd:option('-init_at_local_prediction',1,"whether to init prediction using unary predictor")


	--Loss Options
	cmd:option('-instance_weighted_loss',0,"whether to use an instance-weighted loss. Right now, this is only supported for SRL")
	cmd:option('-loss_type',"log","what training loss to use")
	cmd:option('-negative_example_weight',0.001,"how much to down-weight negative examples (only available for certain losses)")
	cmd:option('-false_positive_penalty',0,"Only for SRL. How much extra weight to place on the loss for arcs that weren't filtered by the preprocessing but have a ground truth null label")



	cmd:option('-results_file',"","optional file base name for writing results files")
	cmd:option('-model_file',"","base name for where to save models. the output .rnn file contains the full unrolled inference network. The .energy_net file only contains the energy network.")
	cmd:option('-out_dir',"./results/","base name for where to save models")


	return cmd

end


--Graveyard of no-longer-supported options
	-- cmd:option('-test_minibatch_size',6400,"batch size at test time")
	-- cmd:option('-print_norms',false,"whether to print the various norms of the parameters after every epoch")
	-- cmd:option('-clone_predictor',0,'whether to use a fixed predictor net, or one that is learned')
	-- cmd:option('-inference_perturbation',0,"whether to add random perturbations to inference optimization during training. currently not fully supported.")
	-- cmd:option('-scale_direct_energy',0,"when clamping the unaries, whether to learn a global scale on them")
	-- cmd:option('-finite_difference_step',0.0001," epsilon used in finite difference approximation of Hessian-vector products")
	--cmd:option('-project',0,'whether to condition on a pretrained initial embedding layer (to speed up feature computation during learning)')
	--cmd:option('-average_loss',0,"whether to use sum_t Loss(y_t,t) rather than just L(y_T,y). Might help things converge faster.")
	--cmd:option('-interp_criterion',0,"only used with average_loss mode. Whether to define the ground truth loss for iterate t to be an interpolation between the ground truth and y_0.")
	--cmd:option('-untied_energy_nets',0,"whether to use untied energy nets at each inference timestep (introduces lots more parameters)")
	--cmd:option('-init_energy',"","where to load energy net from.")


