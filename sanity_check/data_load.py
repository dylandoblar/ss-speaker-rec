# -*- coding: utf-8 -*-

import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset


class VoxCelebClassificationDataset(Dataset):
    def __init__(self, data_path, num_frames, use_pase):
        self.data_path = data_path
        # self.M = M  # number of utterances per speaker
        self.num_frames = num_frames
        self.use_pase = use_pase
        self.speaker_ids = ['voxceleb1/' + spk for spk in os.listdir(os.path.join(data_path, 'voxceleb1'))]
        self.speaker_ids += ['voxceleb1_test/' + spk for spk in os.listdir(
            os.path.join(data_path, 'voxceleb1_test')
        )]
        self.speaker2label = {spkr: lab for lab, spkr in enumerate(self.speaker_ids)}
        self.utt_fnames = {spkr: os.listdir(os.path.join(data_path, spkr)) for spkr in self.speaker_ids}
        self.examples = [(utt_fname, lab) for lab in self.utt_fnames for utt_fname in self.utt_fnames[lab]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        utt_fname, spkr = self.examples[idx]
        label = self.speaker2label[spkr]
        utt_path = os.path.join(self.data_path, spkr, utt_fname)

        utter_feats = np.load(utt_path)
        # randomly sample a window in the utterance of the desired length
        if self.use_pase:
            start_frame = np.random.randint(0, utter_feats.shape[2]-self.num_frames)
            partial_utt = utter_feats[:,:,start_frame:start_frame+self.num_frames]
            partial_utt = np.squeeze(partial_utt)
        else:
            start_frame = np.random.randint(0, utter_feats.shape[1]-self.num_frames)
            partial_utt = utter_feats[:,start_frame:start_frame+self.num_frames]

        # features are the average representation in the window
        utt_rep = np.mean(partial_utt, axis=1)

        return torch.as_tensor(utt_rep, dtype=torch.float), label

