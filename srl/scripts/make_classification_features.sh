
od=processed_data/conll2005

for base in train dev test
do
    
    python scripts/make_classification_data.py SPEN_input/conll2005/${base}_rel_ids_5.txt  SPEN_input/conll2005/role_to_id.txt > $od/$base.labels
    th scripts/merge_classification_data.lua $od/$base.labels SPEN_input/conll2005/${base}_scores.txt $od/$base.classification.torch
done


