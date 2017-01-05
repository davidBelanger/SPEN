
GPUID=0 #set this to something >= 0 to use the GPU

torch_gpu_id=0 #torch always thinks it's running on GPUID 0, but the CUDA_VISIBLE_DEVICES environment variable (set below) changes what that means.
if [ "$GPUID" == -1 ]; then
	torch_gpu_id=-1
fi

d=denoise-runs/`date | sed 's| |_|g'`
log=$d/log.txt
mkdir -p $d

results_file_dir=$d/results
mkdir $results_file_dir

#These options are for restoring stuff from previous runs
#Restore a full unrolled network for gradient-based prediction
#init_full_net=   # for example, $previous_run_dir/model--update_all.50.predictor
#iF="-init_full_net $init_full_net"

#Restore the optimization state of a previous run
#init_opt_state=  # for example, $previous_run_dir/model--update_all.50.opt_state
#iO="-init_opt_state $init_opt_state"

data_options="-train_list denoise/depth/train/train.list -test_list denoise/depth/val/val.list  -out_dir $results_file_dir -model_file $d/model- $iF $iO"
system_options=" -batch_size 10  -gpuid $torch_gpu_id -profile 0"

problem_config=$d/problem-config 
problem_options="-problem_config $problem_config"

#These are the only options that are specific to the problem domain and architecture
inverse_noise_variance=1.0 #the inverse ariance of assumed noise model. The local potentials are multipled by this.
problem_options_str="-use_random_crops 1 -local_term_weight $inverse_noise_variance" #there are other options. they're just using default values for now
th flags/DenoiseOptions.lua $problem_options_str -serialize $problem_config
problem_options="$problem_options -problem Denoise -continuous_outputs 1 "

#There are many other hyperparameters that you may want to play with
inference_options="-init_at_local_prediction 1 -inference_learning_rate 0.1 -max_inference_iters 20  -inference_learning_rate_decay 0 -inference_momentum 0 -learn_inference_hyperparams 0 -unconstrained_iterates 1 -line_search 1 -entropy_weight 0"


general_training_options="-training_method E2E"
base_training_config="-gradient_clip 1.0 -optim_method adam -evaluation_frequency 15 -save_frequency 45  -adam_epsilon 1e-8 -batches_per_epoch 10 -learning_rate_decay 0.0"

# In the current code, the 'features' are just the input image x, and the local classifier just returns the input image x (with no denoising). 
# Therefore, we skip the first two training steps (by setting num_epochs = 0 for them)
pretrain_configs="$base_training_config -learning_rate 0.001 -num_epochs 0 -training_mode pretrain_unaries" 
first_pass_configs="$base_training_config -learning_rate 0.001 -num_epochs 0 -training_mode clamp_features"
second_pass_configs="$base_training_config -learning_rate 0.0005 -num_epochs 500 -training_mode update_all"

# This packages up lua tables for the options that are specific to the different stages of training.
training_config=$d/train-config
echo $pretrain_configs
th flags/TrainingOptions.lua $pretrain_configs -serialize $training_config.0
th flags/TrainingOptions.lua $first_pass_configs -serialize $training_config.1
th flags/TrainingOptions.lua $second_pass_configs -serialize $training_config.2
echo $training_config.{0,1,2} | tr ' ' '\n' > $training_config
training_options="-training_configs $training_config"

cmd="th main.lua $data_options $system_options $problem_options $inference_options $training_options $general_training_options"

echo echo running in $d > $d/cmd.sh   
echo export CUDA_VISIBLE_DEVICES=$GPUID >> $d/cmd.sh                                                                                                                                  
echo $cmd >> $d/cmd.sh           
sh $d/cmd.sh 2>&1 |  tee $log  | tee d-latest.log
