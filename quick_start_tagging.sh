mkdir -p data/sequence
#this makes synthetic data by drawing examples from a ground-truth linear-chain CRF and write it out to a file.

echo Creating Synthetic Train/Test Data by Drawing from a CRF
th test/test_data_serialization_and_loading.lua  

#This fits a new linear-chain CRF to the data generated in the previous step and asserts that the accuracy is not that much less than if you had used the true CRF model that the data was drawn from.
echo
echo Training a CRF on the Data to Test that Obtainable Accuracy Approaches the Accuracy of the True Model
th test/test_chain_crf_mle_from_file.lua 

#This fits a SPEN to the same data. The architecture for sequence tagging is defined in model/ChainSPEN.lua
echo 
echo Training a SPEN on the Data using tag_cmd.sh
sh tag_cmd.sh

