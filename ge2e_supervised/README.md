# Generalized End-to-End Loss for Speaker Verification

PyTorch implementation of speaker ID embedding model and loss function described
in https://arxiv.org/pdf/1710.10467.pdf. Based on the code in
https://github.com/HarryVolek/PyTorch_Speaker_Verification.

This repository contains code to preprocess data and train the model. 
This particular codebase is designed for use with the
[VoxCeleb1 dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).


## Training

To train the speaker embedding model, run `train.py`.

The checkpoint save location, save interval, and hyperparameters such as
batch size can be controlled under the `train` header in
`config/config.yaml`. When training on different types of input features,
the `nfeats` parameter under the `data` header must change to match the
dimension of the feature (e.g. 40 for filterbanks, 256 for PASE+ features).

To train the model using different percentages of the dataset (measured
in terms of number of unique speakers), change the `data_ratio` parameter
under the `train` header (e.g. 1.0 corresponds to 100% of the data,
0.5 corresponds to 50% of the data, etc.).

The script will compute loss and equal error rate (EER) on the validation set
every epoch. Only TI-SV from the paper
(https://arxiv.org/pdf/1710.10467.pdf) is implemented.


