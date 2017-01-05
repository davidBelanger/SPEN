mkdir -p runs
d=runs/`date | sed 's| |_|g'`
log=$d/log.txt
mkdir $d

results_file_dir=$d/results
mkdir $results_file_dir

echo ./data/sequence/crf-data.train > ./data/sequence/crf-data.train.list
echo ./data/sequence/crf-data.test > ./data/sequence/crf-data.test.list

data_options="-train_list ./data/sequence/crf-data.train.list -test_list ./data/sequence/crf-data.test.list  -out_dir $results_file_dir -model_file $d/model- $iC"
system_options=" -batch_size 10  -gpuid -1 -profile 0  -test_minibatch_size 10"

#These are the only options that are specific to the problem domain and architecture
problem_config=$d/problem-config 
problem_options="-problem_config $problem_config"


problem_options_str=""
th flags/SequenceTaggingOptions.lua $problem_options_str -serialize $problem_config
problem_options="$problem_options -problem SequenceTagging"


inference_options="-init_at_local_prediction 1 -inference_learning_rate 0.1 -max_inference_iters 20  -inference_learning_rate_decay 0 -inference_momentum 0.5 -learn_inference_hyperparams 1 -unconstrained_iterates 1"


general_training_options="-training_method E2E"
base_training_config="-gradient_clip 1.0 -optim_method adam -evaluation_frequency 25 -save_frequency 50  -adam_epsilon 1e-8 -gradient_noise_scale 0 \
            -batches_per_epoch 100 -learning_rate_decay 0.0 -learning_rate_decay_start 20 -l2 0 "


pretrain_configs="$base_training_config -learning_rate 0.001 -num_epochs 250 -training_mode pretrain_unaries"
first_pass_configs="$base_training_config -learning_rate 0.001 -num_epochs 100 -training_mode clamp_features"
second_pass_configs="$base_training_config -learning_rate 0.0005 -num_epochs 500 -training_mode update_all"

training_config=$d/train-config
echo $pretrain_configs
th flags/TrainingOptions.lua $pretrain_configs -serialize $training_config.0
th flags/TrainingOptions.lua $first_pass_configs -serialize $training_config.1
th flags/TrainingOptions.lua $second_pass_configs -serialize $training_config.2
echo $training_config.{0,1,2} | tr ' ' '\n' > $training_config
training_options="-training_configs $training_config"

cmd="th main.lua $data_options $system_options $problem_options $inference_options $training_options $general_training_options"

echo echo running in $d > $d/cmd.sh                                                                  
echo $cmd >> $d/cmd.sh                                                                               
sh $d/cmd.sh 2>&1 |  tee $log  | tee latest-run.log

