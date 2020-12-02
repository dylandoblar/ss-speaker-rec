## Unsupervised VoxCeleb trainer

This repository is based on the [`voxceleb_unsupervised`](https://github.com/joonson/voxceleb_unsupervised)
repository for self-supervised speaker verification


#### Dependencies
```
pip install -r requirements.txt
```


#### Implemented models and encoders

```
ResNetSE34L (SAP)
LSTM-3
```

ResNetSE34L is the Fast ResNet-34 model described in the paper
[In defence of metric learning for speaker recognition](https://arxiv.org/abs/2003.11982).
LSTM-3 is a 3-layer LSTM model described in the paper
[Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467).


#### Training example

To train Fast ResNet-34 using pre-extracted log-mel filterbanks as
input features:

```
python3 trainSpeakerNet.py --test_interval 1 --model ResNetSE34L --save_path data/resnet_fbank --train_list lists/voxceleb1_extracted_train_list.txt --test_list lists/voxceleb1_extracted_test_list.txt --train_path data/filterbanks/voxceleb1 --test_path data/filterbanks/voxceleb1_test --max_frames 180 --batch_size 64 --n_feats 40
```

To train Fast ResNet-34 using pre-extracted PASE+ representations as
input features:

```
python3 trainSpeakerNet.py --test_interval 1 --model ResNetSE34L --use_pase True --save_path data/resnet_pase --train_list lists/voxceleb1_extracted_train_list.txt --test_list lists/voxceleb1_extracted_test_list.txt --train_path data/pase_representations/voxceleb1 --test_path data/pase_representations/voxceleb1_test --max_frames 167 --batch_size 64 --n_feats 256
```

To train a model using LSTM-3 as the encoder, replace `ResNetSE34L` with `LSTM`
in the `--model` argument. The arguments can also be passed as
`--config path_to_config.yaml`. Note that the configuration file overrides
the arguments passed via command line.


