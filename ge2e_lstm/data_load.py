#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset


class VoxCelebDataset(Dataset):
    def __init__(self, data_path, M, num_frames, utter_start=0, shuffle=True):
        self.path = data_path
        self.M = M
        self.num_frames = num_frames
        self.file_list = os.listdir(self.path)
        self.shuffle = shuffle
        self.utter_start = utter_start

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        np_file_list = os.listdir(self.path)

        if self.shuffle:
            # select random speaker
            selected_spkr = random.sample(np_file_list, 1)[0]
        else:
            selected_spkr = np_file_list[idx]

        # load utterance spectrograms for selected speaker
        all_utters = np.load(os.path.join(self.path, selected_spkr))

        # select M utterances per speaker
        # dims are ordered as (batch(=M), n_mels(=40), frames)
        if self.shuffle:
            utter_idx = np.random.randint(0, all_utters.shape[0], self.M)
            sampled_utters = all_utters[utter_idx]
        else:
            # utterances of a speaker 
            sampled_utters = all_utters[self.utter_start:self.utter_start+self.M]

        # transpose [batch, frames, n_mels]
        sampled_utters = torch.tensor(np.transpose(sampled_utters, axes=(0,2,1)))
        return sampled_utters

