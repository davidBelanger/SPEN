
# Image Denoising with SPENs

To run a self-contained example of image denoising, cd to the base directory for SPEN, and then execute

`wget https://www.cics.umass.edu/~belanger/depth_denoise.tar.gz`

`tar -xvf depth_denoise.tar.gz`

`depth_cmd.sh`


This is downloads a preprocessed version of a small amount of the depth denoising data from this [paper](http://www.cs.toronto.edu/~slwang/proximalnet.pdf), made available [here](https://bitbucket.org/shenlongwang/), and then fits a SPEN. The associated SPEN architecture is defined in model/DepthSPEN.lua. 

Note that depth denoising isn't a traditional denoising task where we assume a parametric noise model. With a parametric noise model, we can create lots of synthetic training data by corrupting clean images. Here, we simply provide ground truth cleaned images and their corresponding observations from a Kinect sensor. 

Let x be the input blurry image and y be the sharpened image we seek to predict. We recover y by MAP inference, where we find y that maximizes P(x | y ) P(y). We assume a Gaussian noise model, so that P(x|y) is scaled mean squared error. This isn't the correct noise model for the task; the Kinect sensor's errors behave quite differently. However, it is a reasonable assumption. 

There are various parametrizations for the prior distribution P(y). Many previous works have employed a 'field of experts' model: P(y) \propto exp(\sum_i \sum_xy w_i \rho (f_i(x,y))), where f_1(\cdot,\cdot), \ldots, f_k(\cdot,\cdot) are a set of localized linear filter responses and \rho is a nonlinearity. 

Early work estimated the the weights w_i and the filters by maximizing the likelihood of a dataset of sharp images. Inference in the field of experts model is intractable, and thus practitioners employed approximate methods such as contrastive divergence. 

An alternative line of work estimated the parameters using end-to-end approaches, by applying automatic differentiation to the procedure of iteratively solving the MAP objective, for a fixed number of iterations. 

We employ this end-to-end approach, but consider substantially more expressive prior distributions over y than a field of experts from linear filters. Namely, we consider an arbitrary deep network: P(y) \propto exp(D(y)). 


### Related Work
> Stefan Roth and Michael Black. Fields of experts: A framework for learning image priors. In CVPR, 2005.

> Marshall Tappen, Ce Liu, Edward Adelson, and William Freeman. Learning Gaussian Conditional Random Fields for Low-Level Vision. In CVPR, 2007.

> Adrian Barbu. Training an Active Random Field for Real-Time Image Denoising. IEEE Transactions on Image Processing, 2009.

> Kegan Samuel and Marshall Tappen. Learning Optimized MAP Estimates in Continuously-Valued MRF Models. In CVPR, 2009.

> Jian Sun and Marshall Tappen. Learning Non-Local Range Markov Random Field for Image Restoration. In CVPR, 2011.

> Justin Domke. Generic Methods for Optimization-Based Modeling." AISTATS 2012.

### Data Processing
You will need a large number of pairs of noisy and clean images. Some helpful utility code is: 

`scripts/im_pairs_to_torch.lua <file_list> <num examples per file> <output_dir> <num_total_examples>`

This depends on the torch gm package, will requires you to install graphicsmagick. The images can be any format loadable by graphicsmagick.

Each line of file_list is of the form `<blurry_image>\s<sharp image>`
