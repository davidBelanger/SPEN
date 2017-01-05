# Multi-Label Classification with SPENs


To run a self-contained example of multi-label classification, cd to the base directory for SPEN, and then execute

`wget http://www.cics.umass.edu/~belanger/icml_mlc_data.tar.gz`

`tar -xvf icml_mlc_data.tar.gz`

`sh mlc_cmd.sh`

The SPEN architecture for MLC is described in detail in our [paper](https://people.cs.umass.edu/~belanger/belanger_spen_icml.pdf). It is implemented in MLCSPEN.lua. See main.lua for the load_problem implementation for MLC. This also instantiates data loading, evaluation, etc.

We evaluate using evaluate/MultiLabelEvaluation.lua, which computes F1 score. This depends on a threshold, between 0 and 1, for converting soft decisions to hard decisions. If you use the -predictionThresh argument (eg., when evaluating on your test set), then we use a single threshold. Otherwise, it tries a bunch of thresholds and finds the best F1.

Note that our new code does not reproduce the configuration of the ICML experiments. The evaluation is the same, but the training method is substantially different. Even if you train with an SSVM loss, there are various configuration differences (eg. how we detect convergence of the inner prediction problem) 

### Data Processing
For new data it will be useful to use the conversion script

`scripts/ml2torch.lua <features_file> <labels_file> <out_file>`

Here, features_file contains one example per line, where each line is a space-separated list of feature values. The labels_file is defined similarly. For some problems, the labels and/or features are sparse. However, you'll need to provide them as a dense vector here (the SPEN code assumes they're dense anyway). You'll want to use this conversion script for both the train and test data. 
