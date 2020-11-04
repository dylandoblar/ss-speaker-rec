#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset


class VoxCelebDataset(Dataset):
    def __init__(self, data_path, M, num_frames, shuffle=True):
        self.data_path = data_path
        self.M = M
        self.num_frames = num_frames
        self.file_list = os.listdir(self.data_path)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        self.file_list = os.listdir(self.data_path)

        if self.shuffle:
            # select random speaker
            selected_spkr = random.sample(self.file_list, 1)[0]
        else:
            selected_spkr = self.file_list[idx]

        # randomly select M utterance files per speaker
        path_to_spkr_utts = os.path.join(self.data_path, selected_spkr)
        sampled_utters = random.sample(path_to_spkr_utts, M)

        # load in the M selected utterances
        # dims are ordered as (n_feats (n_mels or pase_dim), frames)
        selected_utters = []
        for full_utter in sampled_utters:
            utter_spec = np.load(os.path.join(path_to_spkr_utts, full_utter))
            start_frame = np.random.randint(0, utter_spec.shape[1]-self.num_frames)
            partial_utt = utter_spec[:,start_frame:start_frame+self.num_frames]
            selected_utters.append(partial_utt)
        selected_utters = np.stack(selected_utters, axis=0)  # (batch, n_feats, frames)

        # transpose to (batch, frames, n_feats)
        selected_utters = torch.tensor(np.transpose(selected_utters, axes=(0,2,1)))
        return selected_utters





