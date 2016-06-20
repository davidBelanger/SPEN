
outdir=ml-logs
d=$outdir/`date | sed 's| |_|g'`
log=$d/log.txt
mkdir -p $d

#if you want to initialize the features / unary potentials from something pretrained, specify it here
#initClassifier=XXX
#iF="-initFeaturesNet $initClassifier"

resultsFileDir=$d/results
mkdir $resultsFileDir
dataOptions="-trainList XXX -testList XXX  -outDir $resultsFileDir -modelFile $d/model- -inputSize 1836 -labelDim 159 $iF"


inferenceOptions="-inferenceLearningRate 0.1 -maxInferenceIters 20  -inferenceLearningRateDecay 0 -inferenceMomentum 0.5 -averageLoss 0 -learnInferenceHyperparams 1 -unconstrainedIterates 1"

architectureOptions="-initAtLocalPrediction 1 -finalFeatureDim 25 -energyDim 16 -energyNonlinearity SoftPlus  -featuresNonlinearity ReLU -positiveWeight 0 -labelEnergy 1   -directEnergy 1 -conditionalLabelEnergy 0 -labelEnergyType deeper -linearFeatures 0   -dropout 0.25 -embeddingDim 150 -featureDim 125 -initUnaryWeight 0.1"

baseTrainingConfig="-gradientClip 1.0 -optimMethod adam -evaluationFrequency 25 -saveFrequency 25  -adamEpsilon 1e-8 -gradientNoiseScale 0 \
            -batchesPerEpoch 100 -learningRateDecay 0.0 -learningRateDecayStart 20 -l2 0 "

problemOptions=" -lossType log -problem ML "
systemOptions=" -minibatch 32  -gpuid -1 -profile 0  -testMinibatchSize 5"

pretrainConfigs="$baseTrainingConfig -learningRate 0.0001 -numEpochs 10 -trainingMode pretrainUnaries"
firstPassConfigs="$baseTrainingConfig -learningRate 0.0001 -numEpochs 100 -trainingMode clampFeatures -learnUnaryInFirstPass 0"
secondPassConfigs="$baseTrainingConfig -learningRate 0.00005 -numEpochs 500 -trainingMode updateAll"

#todo: using mktemp won't work if you're qsubbing
trainingConfig=$d/config
echo $pretrainConfigs
th TrainingOptions.lua $pretrainConfigs -serialize $trainingConfig.0
th TrainingOptions.lua $firstPassConfigs -serialize $trainingConfig.1
th TrainingOptions.lua $secondPassConfigs -serialize $trainingConfig.2
echo $trainingConfig.{0,1,2} | tr ' ' '\n' > $trainingConfig
trainingOptions="-trainingConfigs $trainingConfig"


cmd="th main.lua $systemOptions $problemOptions $inferenceOptions $architectureOptions $dataOptions $trainingOptions"


echo echo running in $d > $d/cmd.sh                                                                  
echo $cmd >> $d/cmd.sh                                                                               
sh $d/cmd.sh 2>&1 |  tee $log  | tee b-latest.out


