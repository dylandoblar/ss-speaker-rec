# Multi-way Speaker Classification

This directory contains the code necessary to train a simple model on a multi-way
speaker classification task.  There are two models implemented: a linear model and
a simple single-hidden-layer feedforward network with ReLU nonlinearity.  The purpose
of this task is to characterize how efficiently and accessibly speaker identity information
is encoded in PASE+ features as compared to log-mel filterbanks.


## Data preprocessing

Pre-extract log-mel filterbanks and PASE+ features by following the instructions in the
Data section of the [top-level README](../README.md#data);
detailed instructions for filterbanks can be found in the Data preprocessing section of the 
[ge2e_supervised README](../ge2e_supervised/README.md#data-preprocessing).

## Training

To train a model on the multi-way speaker classification task, run `train.py`.

The top-level data directory can be configured by setting the `data_path` variable
in the script, and the logging directory can be specified by the `log_dir` variable.
The training and validation loss and accuracy are written to a CSV file in the
directory specified by `log_dir`.
By default, 167 frames are used if PASE+ features are selected, and 180 frames are used
if filterbanks are selected to maintain a receptive field of ~1.8 seconds in each case.
The model type can be set using the `model_type` parameter: `'linear'` will use the
linear model and `'hidden'` will use the model with a single hidden nonlinear layer.

## Plots

To plot the training and validation loss and accuracy, run the `plot.py` script.
Specify the name of the experiment to generate plots for with the `exp_name` variable,
and the directory to store the outputs with the `save_dir` variable.
