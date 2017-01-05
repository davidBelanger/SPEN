# Structured Prediction Energy Network Training Code

Structured Prediction Energy Networks (SPENs) are a flexible, expressive approach to structured prediction. See our paper:

[David Belanger](https://people.cs.umass.edu/~belanger/) and [Andrew McCallum](https://people.cs.umass.edu/~mccallum/pubs.html) "Structured Prediction Energy Networks." ICML 2016. [link](https://people.cs.umass.edu/~belanger/belanger_spen_icml.pdf)


<!-- This project contains [torch](http://torch.ch/) code for SPENs. We provide code for two use cases: multi-label classification and image denoising. We also provide a generic API for which it should be easy to prototype additional applications. If you would like to do so, feel free to contact David Belanger for advice.  -->

## Updates in Version 2
Basically everything. The code is substantially more modular: it now provides proper abstractions between models, prediction methods, training losses, etc. We have also added a considerable number of tests. We have also added back a structured SVM training method, as was used in the ICML paper, and examples for sequence tagging. Algorithmically, there are a number of improvements, including backpropagation through a broader selection of optimization methods, dynamic unrolling of the computation graph for iterative prediction (to account for variable numbers of iterations), and explicit regularization to encourage the iterative prediction to converge quickly. 

Note that some functionality, such as dropout or different batch sizes at test time vs. train time, is no longer supported in this code. Some, but not all of it could be added back easily. Let us know if you have particular requests. 

## Differences Between this Code and the ICML Paper code

The ICML paper trains the energy network using a structured SVM (SSVM) loss. As we discuss in the paper, this approach does not gracefully handle situations where inexact optimization is performed in the inner loop of training. Since our energy functions are non-convex with respect to the output labels, this is a key concern in both in theory and practice. In response, we have recently switched to more straightforward, 'end-to-end' training approach, based on:

[Justin Domke](http://users.cecs.anu.edu.au/~jdomke/)  "Generic Methods for Optimization-Based Modeling." AISTATS 2012. [link](http://www.jmlr.org/proceedings/papers/v22/domke12/domke12.pdf).

Here, we construct a long computation graph corresponding to running gradient descent on the energy function for a fixed number of iterations. With this, prediction amounts to a feed-forward pass through this recurrent neural network, and training can be performed using backprop. There are some technical details regarding how to backpropagate through the process of taking gradient steps, and we employ Domke's finite differences technique. The advantage of this end-to-end approach is that we directly optimize the empirical loss: the computation graph used at train time is an exact implementation of the gradient-based inference (for a fixed number of steps) that we use at test time. 

The only restriction on the family of energy functions optimizable with this approach vs. the structured SVM approach is that we need our energy function to be smooth (with respect to both the parameters and the inputs). Rather than using ReLUs, we recommend using a SoftPlus. An ironic downside of the end-to-end approach fitting the training data much better is that it is more prone to overfitting. Therefore, it does not necessarily generate better test performance on the relatively small multi-label classification datasets we considered in the ICML paper.

## Useful Library Code
We provide various bits of stand-alone code that might be useful in other applications. See their respective files for documentation. 

`optimize/UnrolledGradientOptimizer.lua` takes an energy network E(y,x), and a network for guessing an initial value y0, and constructs a recurrent neural network that performs gradient-based minimization of the energy with respect to y. It provides various options for doing gradient descent with momentum, line search, etc.

`optimize/GradientDirection.lua` takes an energy network E(y,x) and returns an nn module that returns the gradient of the energy with respect to y in the forward pass. In the backwards pass, the Hessian-vector product is computed using finite differences. 

`infer1d/*.lua and model/ChainCRF.lua` provide various useful code for inference and learning in linear-chain CRFs. See various tests for examples of how to use these. 

## Applications
We are releasing code for three applications: [Multi-Label Classification](MultiLabelClassification.md), [Sequence Tagging](Tagging.md), and [Image Denoising](Denoising.md). All of these contain quick start scripts. 

It is straightforward to implement new structured prediction applications using our code. See our [API](Applications.md) documentation.

## Quick Start 

We recommend running the sequence tagging example `quick_start_tagging.sh`. This uses main.lua, which has lots of functionality. For a more simple example, you can use test/test_chain_spen_learn.lua. 

## Code Dependencies
You'll need to install the following torch packages, which can all be installed using 'luarocks install X:' torch, nn, cutorch, cunn, optim, nngraph. If you're doing stuff with images, we recommend configuring cudnn and using the -cudnn flag to main.lua.

Finally, we use various utility functions from David's [torch-util](https://github.com/davidBelanger/torch-util) project. You will need to clone torch-util such that its relative path to this project is ../torch-util. 


## Options
See ./flags/*.lua for the various command line options and their explanations. See the example applications described above to see how some of the flags have been used. 
