
od=processed_data/conll2005

for base in dev train test
do
    python scripts/process_arcs_with_collisions.py SPEN_input2/${base}_rel_ids_5.txt  SPEN_input2/role_to_id.txt  $od/$base.int

    th scripts/arcs2torch.lua $od/$base.int.arcs $od/$base.arcs.torch
    for ext in p2p a2a p2a
    do
    	th scripts/collisions2torch.lua $od/$base.int.$ext $od/$base.collisions.$ext
    done
#    th ../scripts/csv2torch.lua SPEN_input2/${base}_scores.txt $od/$base.features.torch
done


