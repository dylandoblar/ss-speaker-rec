# Multi-task Learning of Speech Representations for Speaker Recognition

This repository contains code for a class project for MIT's Meta Learning
class (6.883), Fall 2020.

We compare the effectiveness of learned representations from the
[problem-agnostic speech encoder (PASE+)](https://arxiv.org/abs/2001.09239)
with log-mel filterbanks when used for the task of speaker recognition.
Specifically, we use the two types of features as as inputs to various
speaker embedding models and compare the resulting performance.


## Baseline code

The baseline code for the `pase`, `ge2e_supervised`, and `self_supervised`
directories is from the [`pase`](https://github.com/santi-pdp/pase),
[`PyTorch_Speaker_Verification`](https://github.com/HarryVolek/PyTorch_Speaker_Verification), and
[`voxceleb_unsupervised`](https://github.com/joonson/voxceleb_unsupervised)
repositories, respectively, and was modified for our experiments.


## Data

All experiments were done using the VoxCeleb1 dataset. To download and extract
the data, follow the directions on the
[VoxCeleb website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).
Alternatively, you can install the dependencies and follow the appropriate
data preparation instructions in the
[`voxceleb_trainer`](https://github.com/clovaai/voxceleb_trainer) repository
from Clova AI.

All data loaders depend on having pre-extracted features (log-mel filterbanks
or PASE+ features). To preprocess data in this way, run
`ge2e_supervised/extract_filterbanks` or `pase/extract_pase_features`,
respectively.

To run these scripts, the data must be organized in the following
directory structure:

```
/voxceleb1_wav/speaker_id/utterance_id.wav
```

For example, running

```
python3 extract_filterbanks.py voxceleb1_wav voxceleb1_fbank
```

will process the utterances in voxceleb1_wav and save npy files in the
following structure:

```
/voxceleb1_fbank/speaker_id/00001.npy
/voxceleb1_fbank/speaker_id/00002.npy
...
```

Likewise, running
```
python3 extract_pase_features.py voxceleb1_wav voxceleb1_pase
```

will save npy files in the following directory structure:

```
/voxceleb1_pase/speaker_id/00001.npy
/voxceleb1_pase/speaker_id/00002.npy
...
```

Voice activity detection (VAD) can optionally be done as part of extraction
(using methods such as Google's WebRTCVAD, for example), but we do not do so
because we assume that utterances in VoxCeleb1 consist of mostly speech.


## Experiments

Refer to the READMEs of each of the subdirectories for more details on how to
run experiments with various settings.

