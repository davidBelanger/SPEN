# Structured Prediction Energy Network Training Code

Structured Prediction Energy Networks (SPENs) are a flexible, expressive approach to structured prediction. See our paper:

[David Belanger](https://people.cs.umass.edu/~belanger/) and [Andrew McCallum](https://people.cs.umass.edu/~mccallum/pubs.html) "Structured Prediction Energy Networks." ICML 2016. [link](https://people.cs.umass.edu/~belanger/belanger_spen_icml.pdf)


This project contains [torch](http://torch.ch/) code for SPENs. We provide code for two use cases: multi-label classification and image denoising. We also provide a generic API for which it should be easy to prototype additional applications. If you would like to do so, feel free to contact David Belanger for advice. 


## New End-to-End Training Method
The ICML paper trains the energy network using a structured SVM (SSVM) loss. As we discuss in the paper, this approach does not gracefully handle situations where inexact optimization is performed in the inner loop of training. Since our energy functions are non-convex with respect to the output labels, this is a key concern in both in theory and practice. 

In response, we have recently switched to more straightforward, 'end-to-end' training approach, based on:

[Justin Domke](http://users.cecs.anu.edu.au/~jdomke/)  "Generic Methods for Optimization-Based Modeling." AISTATS 2012. [link](http://www.jmlr.org/proceedings/papers/v22/domke12/domke12.pdf).

Here, we construct a long computation graph corresponding to running gradient descent on the energy function for a fixed number of iterations. With this, prediction amounts to a feed-forward pass through this recurrent neural network, and training can be performed using backprop. There are some technical details regarding how to backpropagate through the process of taking gradient steps, and we employ Domke's finite differences technique. The advantage of this end-to-end approach is that we directly optimize the empirical loss: the computation graph used at train time is an exact implementation of the gradient-based inference (for a fixed number of steps) that we use at test time. 

The only restriction on the family of energy functions optimizable with this approach vs. the structured SVM approach is that we need our energy function to be smooth (with respect to both the parameters and the inputs). Rather than using ReLUs, we recommend using a SoftPlus approximation. 

Our end-to-end approach is much more simple code-wise than the approach in the ICML paper and is less sensitive to hyperparameters. For example, the SSVM is very sensitive to stopping criteria for the inner optimization problem. End-to-end training also produces substantially better training losses on our multi-label classification data.  In response, we are not releasing code for SSVM training. An ironic downside of the end-to-end approach fitting the training data much better is that it is more prone to overfitting. Therefore, it does not necessarily generate better test performance on the relatively small multi-label classification datasets we considered in the ICML paper.

## Code Dependencies
You'll need to install the following torch packages, which can all be installed using 'luarocks install X:' torch, nn, cutorch, cunn, optim, nngraph. 

The 'deep mean-field' part of the code also depends on the autograd package. If you're doing stuff with images, we recommend configuring cudnn and using the -cudnn flag to main.lua.

Finally, we use various utility functions from David's [torch-util](https://github.com/davidBelanger/torch-util) project. You will need to clone torch-util such that its relative path to this project is ../torch-util. 

## Applications
We are releasing code for two applications: [Multi-Label Classification](MultiLabelClassification.md) and [Image Denoising](Denoising.md).	

It is straightforward to implement new structured prediction applications using our code. See our [API](Applications.md) documentation.

## Options
See the top of main.lua for a long list of command line options and their explanations. 


## Coding Style
Some time this year I adopted a terrible habit of interweaving camelCase and separated\_by\_underscores coding styles. I apologize. I will fix this at some point. 
