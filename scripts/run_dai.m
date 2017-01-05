clear all;
addpath(genpath('~/TT-MRF/'));
addpath(genpath('~/libDAI-0.3.2/matlab/'));


n = 10
m = 20
temperature = 1.0
Model = generate_spin_glass_model(n, m, temperature);
dai(Model.libdaiFactors,'BP')