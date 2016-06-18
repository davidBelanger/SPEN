# The SPENProblem API

SPEN applications, such as SPENMultilabelClassification and SPENDenoise extend the SPENProblem class. You will need to implement the abstract methods defined towards the top of SPENProblem.lua for data loading, preprocessing, evaluation, etc. You will also need to make sure that your new class contains the following members:

`problem.inference_net`: the energy network E_x(y), but using pre-computed features rather than the raw value of x. This takes {labels, features} and returns a single number per minibatch element

`problem.fixed_features_net`: Feature mapping F(x). This may be pretrained using classification, or loaded from file. If the training mode is 'clampFeatures', then we don't update its parameters, and don't even backprop through it during training. 
 
`problem.learned_features_net`: (Optional) The overall feature mapping is fixed_features_net followed by learned_features_net. These features are learned even in 'clampFeatures' mode.

`problem.initialization_net`: This network takes {x,F(x)} and returns an initial guess y_0 for the labels. The reason it takes the raw input x is that this might be important for getting the size of the inputs when the problem can have variable-sized inputs. See SPENProblem.lua to see how this interacts with the --initAtLocalPrediction flag. Generally, you don't implement initialization_net directly. Instead, you decide to init the labels with the outputs of a local classifier, or initialize them to some fixed hard-coded value (eg. 0).

`problem.iterate_transform`: problem.iterate_transform --Everything is set up for unconstrained optimization. This maps things onto the constrain set at the end of optimization. For example, it converts logits to probabilities. Set to Identity() if you don't need a transformation. 



`problem.iterateRange`: A table of 2 numbers. If we are doing projected gradient descent for test-time optimization, this is the upper and lower bound on the value for each prediction variable, eg. {0,1} or {0,255}.

`problem.input_features_pretraining_net`: This is a simple feed-forward classifier, often used for pretraining the features used by the inference_net. Also, for problems where the energy function has terms analogous to the 'unary potentials' of a graphical model, this classifier may provide these per-label terms. 

`problem.structured_training_loss.loss_criterion`: The training criterion. Used for pretraining the 'unaries' and also for the RNN net. 

###Block-Structured Y
For some problems, there are multiple optimization variables, but downstream you only care about one of them. For example, in blind deconvolution we have a latent image and latent blur kernel and we often only care about the image. We provide limited support for this. Depending on the problem structure, you may want to do something different than joint gradient descent on all variables, eg. block coordinate descent. Here, we assume that y is a table of optimization variables. 

`problem.prediction_selector`:  This grabs that one from a table of optimization variables. SPENProblem has a default Identity() implementation. 

`problem.learning_rate_multiplier_per_block`: Table of learning rates to use for table element of y. 

`problem.numOptimizationVariables`: Size of the y table. 

`problem.alternatingUpdates`: Whether to round-robin gradient descent on each element of y, rather than stepping on all of them at once. Basically, poor-man's block coordinate descent. 


