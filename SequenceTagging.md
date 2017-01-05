# Sequence Tagging with SPENs

For a self-contained example of how to do sequence tagging with SPENs, see:

`quick_start_tagging.sh`

This first generates some synthetic data and then fits a SPEN to it. The actual training is called from tag_cmd.sh. 

See model/ChainSPEN.lua for the architecture. It is very simplistic, and closely matches the energy function defined by a first-order linear-chain CRF. It would be easy to make the energy more expressive. 

Also, note that we provide lots of utility code for linear-chain conditional CRFs, including maximum likelihood learning, mean-field inference, sum-product inference, etc. These were used to test SPENs. See various tests for how the CRF stuff is used. 


## Data Preprocessing
The tag_cmd.sh script expects a data file that was saved with torch.save() and contains a single table with two fields:

data: Tensor of shape num_examples x sequence_length x feature_dim

labels: Tensor of shape num_examples x sequence_length x 1

(you should inspect the files saved by test/test_data_serialization_and_loading.lua to see what these look like. Just open up a command line lua session by calling 'th' and then load them using torch.load.)  

See test/test_csv_io.lua for some examples of utility code for reading and writing csv files. For sequence tagging, your input features will probably be three-dimensional (since each time point has multiple features associated with it). You'll need to write to write some new code to translate these into a torch Tensor. Let me know if you have questions.

