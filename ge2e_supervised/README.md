# Generalized End-to-End Loss for Speaker Verification

PyTorch implementation of speaker ID embedding model and loss function described
in https://arxiv.org/pdf/1710.10467.pdf. Based on the code in
https://github.com/HarryVolek/PyTorch_Speaker_Verification.

This repository contains code to preprocess data and train the model. 
This particular codebase is designed for use with the
[VoxCeleb1 dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).


## Data preprocessing

Any dataset must be organized in the following directory structure to be
preprocessed:

```
/base_directory/speaker_id/utterance_id
```

For each speaker ID, running `extract_filterbanks` computes the
log-mel filterbanks for all audio utterances in the base directory and
saves them in npy format in the specified output directory.
For example, running

```
python3 extract_filterbanks.py voxceleb1_dev_wav voxceleb1_dev_npy
```

will process the utterances in voxceleb11_dev_wav and save npy files in the
following structure:

```
/voxceleb1_dev_npy/speaker_id/00001.npy
/voxceleb1_dev_npy/speaker_id/00002.npy
...
```


## Training

The training pipeline is based heavily on the code by Harry Volek in
https://github.com/HarryVolek/PyTorch_Speaker_Verification.

To train the speaker embedding model, run `train.py`.

The checkpoint save location, save interval, and hyperparameters such as
batch size can be controlled under the `train` header in
`config/config.yaml`. When training on different types of input features,
the `nfeats` parameter under the `data` header in the config file
must change to match the dimension of the feature (e.g. 40 for filterbanks,
256 for PASE+ features).

To train the model using different percentages of the dataset (measured
in terms of number of unique speakers), change the `data_ratio` parameter
under the `train` header (e.g. 1.0 corresponds to 100% of the data,
0.5 corresponds to 50% of the data, etc.).

The script will compute loss and equal error rate (EER) on the validation set
every epoch. Only TI-SV from the paper
(https://arxiv.org/pdf/1710.10467.pdf) is implemented.


