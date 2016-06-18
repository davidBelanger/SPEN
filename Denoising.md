
# Image Denoising with SPENs

A SPEN architecture for Image Denoising is implemented in Denoising.lua, with some general functionality added in SPENProblem.lua. 

Let x be the input blurry image and y be the sharpened image we seek to predict. We recover y by MAP inference, where we find y that maximizes P(x | y ) P(y). We assume a Gaussian noise model, so that P(x|y) is scaled mean squared error. There are various parametrizations for the prior distribution P(y). Many previous works have employed a 'field of experts' model: P(y) \propto exp(\sum_i \sum_xy w_i \rho (f_i(x,y))), where f_1(\cdot,\cdot), \ldots, f_k(\cdot,\cdot) are a set of localized linear filter responses and \rho is a nonlinearity. 

Early work estimated the the weights w_i and the filters by maximizing the likelihood of a dataset of sharp images. Inference in the field of experts model is intractable, and thus practitioners employed approximate methods such as contrastive divergence. 

An alternative line of work estimated the parameters using end-to-end approaches, by applying automatic differentiation to the procedure of iteratively solving the MAP objective, for a fixed number of iterations. 



<!--- #"Generic Methods for Optimization-Based Modeling." AISTATS 2012. [link](http://www.jmlr.org/proceedings/papers/v22/domke12/domke12.pdf). --->

We employ this end-to-end approach, but consider substantially more expressive prior distributions over y than a field of experts from linear filters. Namely, we consider an arbitrary deep network: P(y) \propto exp(D(y)). We also support functionality where D can have terms that operate in the frequency domain. 


### Data Processing
You will need a large number of sharp images, which you can then add noise to using some simple code. The denoising code assumes a Gaussian likelihood, so to avoid model mis-specification you should add white noise. However, feel free to use alternative image corruptions. Some helpful utility code is: 

`scripts/im_pairs_to_torch.lua <file_list> <num examples per file> <output_dir> <num_total_examples>`

This depends on the torch gm package, will requires you to install graphicsmagick. The images can be any format loadable by graphicsmagick.

Each line of file_list is of the form `<blurry_image>\s<sharp image>`



### Related Work
> Stefan Roth and Michael Black. Fields of experts: A framework for learning image priors. In CVPR, 2005.

> Marshall Tappen, Ce Liu, Edward Adelson, and William Freeman. Learning Gaussian Conditional Random Fields for Low-Level Vision. In CVPR, 2007.

> Adrian Barbu. Training an Active Random Field for Real-Time Image Denoising. IEEE Transactions on Image Processing, 2009.

> Kegan Samuel and Marshall Tappen. Learning Optimized MAP Estimates in Continuously-Valued MRF Models. In CVPR, 2009.

> Jian Sun and Marshall Tappen. Learning Non-Local Range Markov Random Field for Image Restoration. In CVPR, 2011.

> Justin Domke. Generic Methods for Optimization-Based Modeling." AISTATS 2012.

### Applications

Besides providing an effective image denoising network, this learning procedure produces a standalong network P(y), which returns the prior log-probability of a given image. This may be useful in various downstream tasks. You could even sample from the space of images using, for example, Hamiltonian Monte Carlo. 
