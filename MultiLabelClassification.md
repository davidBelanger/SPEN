# Multi-Label Classification with SPENs

The SPEN architecture for MLC is described in detail in our [paper](https://people.cs.umass.edu/~belanger/belanger_spen_icml.pdf). It is implemented in SPENMultiLabelClassification.lua, with some general functionality added in SPENProblem.lua. See main.lua for a description of the command line arguments. 


SPENMultiLabelClassification calls MultiLabelEvaluation.lua, which computes F1 score. This depends on a threshold, between 0 and 1, for converting soft decisions to hard decisions. If you use the -predictionThresh argument (eg., when evaluating on your test set), then we use a single threshold. Otherwise, it tries a bunch of thresholds and finds the best F1.


See ml_cmd.sh for an example script for running the code.

### Data Processing
It will be useful to use the conversion script

`scripts/ml2torch.lua <features_file> <labels_file> <out_file>`

Here, features_file contains one example per line, where each line is a space-separated list of feature values. The labels_file is defined similarly. For some problems, the labels and/or features are sparse. However, you'll need to provide them as a dense vector here (the SPEN code assumes they're dense anyway). You'll want to use this conversion script for both the train and test data. 
