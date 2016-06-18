log_dir=XXX
d=XXX/`date | sed 's| |_|g'`
log=$d/log.txt
mkdir $d

resultsFileDir=$d/results
mkdir $resultsFileDir
dataOptions="-trainList XXX -testList XXX  -outDir $resultsFileDir -modelFile $d/model-"

#use this sort of thing if you want to resume an existing training job
#modelOptions="-initFullNet <prev_expt_dir>/model--updateAll-400.model"
#modelOptions2=" -initOptState <prev_expt_dir>/model--updateAll-500.optState"

mkdir $d/ims
analysisOptions="-writeImageExamples 50 -imageExampleDir $d/ims/ -resultsFile $d/testIms"

inferenceOptions="-inferenceLearningRate 1.0  -maxInferenceIters 10  -inferenceLearningRateDecay 0 -inferenceMomentum 0 -averageLoss 0 -learnInferenceHyperparams 0 -unconstrainedIterates 1"

architectureOptions="-initAtLocalPrediction 1 -finalFeatureDim 25 -energyDim 25 -energyNonlinearity SoftPlus  -featuresNonlinearity ReLU  -linearFeatures 1   -dropout 0 -embeddingDim 8 -featureDim 100"

baseTrainingConfig="-gradientClip 1.0 -optimMethod adam -evaluationFrequency 50 -saveFrequency 100  -adamEpsilon 1e-8 -gradientNoiseScale 0 \
            -batchesPerEpoch 15 -learningRateDecay 0.0 -learningRateDecayStart 20 -l2 0 "

problemOptions=" -lossType mse -problem Denoise "
systemOptions=" -minibatch 32  -gpuid 0 -profile 0  -testMinibatchSize 5"

pretrainConfigs="$baseTrainingConfig -learningRate 0.001 -numEpochs 0 -trainingMode pretrainUnaries"
firstPassConfigs="$baseTrainingConfig -learningRate 0.001 -numEpochs 0 -trainingMode clampFeatures -learnUnaryInFirstPass 0"
secondPassConfigs="$baseTrainingConfig -learningRate 0.005 -numEpochs 1000 -trainingMode updateAll $modelOptions2"

#todo: using mktemp won't work if you're qsubbing
trainingConfig=$d/config
echo $pretrainConfigs
th TrainingOptions.lua $pretrainConfigs -serialize $trainingConfig.0
th TrainingOptions.lua $firstPassConfigs -serialize $trainingConfig.1
th TrainingOptions.lua $secondPassConfigs -serialize $trainingConfig.2
echo $trainingConfig.{0,1,2} | tr ' ' '\n' > $trainingConfig
trainingOptions="-trainingConfigs $trainingConfig"


cmd="th main.lua $systemOptions $problemOptions $inferenceOptions $architectureOptions $dataOptions $trainingOptions $analysisOptions $modelOptions"


echo echo running in $d > $d/cmd.sh                                                                  
echo $cmd >> $d/cmd.sh                                                                               
rm -f pRun
ln -s $d ./pRun
sh $d/cmd.sh 2>&1 |  tee $log | tee p-latest.out


