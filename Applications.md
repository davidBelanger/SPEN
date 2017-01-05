# Implementing New SPEN Applications

See main.lua for examples of various SPEN applications. The SPEN code is quite modular. The only thing that needs to be implemented is the load_problem method that is build in main.lua. This returns the following application-specific items.

`model` is an object that obeys the SPEN api, described below.

`y_shape` is a table containing the shape of y, the iterates for gradient-based SPEN optimization.

`evaluator_factory` is a function that takes a batcher and a soft predictor and returns an object that implements an evaluate(timestep) method used for evaluating and logging performance. 

`preprocess_func` is a function that takes (y,x,num_examples) and returns optionally preprocessed versions of the data (eg. expanding int indices for y to a one-hot representation.) If preprocess_func is nil, then no such transformation will be applied. 

`train_batcher` Object that provides two methods: get_iterator() and get_ongoing_iterator(). The first is an iterator used typically used for test data, which returns {nil,nil,nil} when it reaches the end of the data. The second is an infinite iterator (eg., it loops back to the beginning of the dataset once it reaches the end). Each method returns a lua iterator that can be called and will return: {y,x, num_actual_examples}. The outer dimension for y and x is expected to be params.batchsize always. If there isn't enough data to fill a tensor of this size, it may zero-pad the data, in which case num_actual_examples refers to the number of actual examples. This is useful at test time to make sure that over-inflated accuracy numbers are not computed on the padding. 

`test_batcher` Similar batcher, but for test data. 


## The SPEN API

SPEN applications, extend the SPEN class, given in model/SPEN.lua. See model/ for various examples.


### Methods that SPEN Subclasses Must Implement 

`SPEN:features_net()` returns a network that takes in x and returns features F(x)

`SPEN:unary_energy_net()` returns a network that takes F(x) and returns a set of 'local potentials' such that the local energy term is given between the inner product between the output of this network and self:convert_y_for_local_potentials(y). This network is used as a term in the SPEN energy. It is also used as the local classifier used in pretraining the features and optionally as a means to initialize a guess for y0 when performing iterative optimization to form predictions.

`SPEN:convert_y_for_local_potentials(y)` takes an nngraph Node and returns an nngraph node used for taking inner products with the 'local potentials' of the unary energy network. Typically, this can be set to the identity.

`SPEN:global_energy_net()` return a network that takes {y,F(x)} and returns a number. The total SPEN energy is the sum of this and the unary_energy_net, where the global term is weighted by config.global_term_weight. 

`SPEN:uniform_initialization_net()` returns a network that takes no inputs and returns an initial guess y0 for iterative optimization. A default implementation, using nn.Constant, is implemented in SPEN.lua. Only override this if necessary. 

## SPEN Members and Methods that Outside Code Accesses
`spen.initialization_network` takes x and returns a guess for y for iterative optimization. May or may not be the same as the classifier network. 

`spen.features_network` takes x and returns features F(x)

`spen.energy_network` takes {y,F(x)} and returns an energy value (to be minimized wrt y)

`spen.classifier_network` takes x and returns a guess for y. This is used for pretraining. 

`spen.global_potentials_network` takes {y,F(x)} and returns the value of the global energy terms.

`spen:set_feature_backprop(value)` takes a boolean value. If value is true, then no backprop will be performed through the features network during training. This prevents the parameters of the features network from being updated. 

`spen:set_unary_backprop(value)` Similarly, this prevents any updates to both the features network and the local potentials, which are a term in the energy function, and also may be used for the initialization_network. 

### Config options for SPEN

The SPEN construct takes two tables, config and params, where the first is for application-specific options and the second contains general options for the entire SPEN software package. 
`params.use_cuda` whether to use the GPU. 

`params.use_cudnn` whether to use cudnn implementations for certain nn layers. 

`params.init_at_local_prediction` whether gradient-based prediction should initialize y0 uniformly or using the local classifier network. 

`config.y_shape` a table for the shape of y, the input to the energy function.

`config.logit_iterates` whether gradient-based optimization of the energy is done in logit space or in normalized space.

`config.global_term_weight` weight to place on global energy terms vs. local energy terms when composing the full energy function. 


