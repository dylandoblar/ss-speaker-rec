#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset


class VoxCelebDataset(Dataset):
    def __init__(self, data_path, M, num_frames, use_pase, data_ratio=1.0, shuffle=True):
        self.data_path = data_path
        self.M = M
        self.num_frames = num_frames
        self.speaker_ids = sorted(os.listdir(self.data_path))
        self.use_pase = use_pase
        self.data_ratio = data_ratio
        self.shuffle = shuffle

        if self.data_ratio != 1.0:
            end_idx = int(len(self.speaker_ids)*data_ratio)
            self.speaker_ids = self.speaker_ids[:end_idx]

        if self.shuffle:
            shuffle(self.speaker_ids)

    def __len__(self):
        return len(self.speaker_ids)

    def __getitem__(self, idx):
        selected_spkr = self.speaker_ids[idx]

        # randomly select M utterance files per speaker
        path_to_spkr_utts = os.path.join(self.data_path, selected_spkr)
        utter_list = os.listdir(path_to_spkr_utts)
        sampled_utters = random.sample(utter_list, self.M)

        # load in the M selected utterances
        # PASE features are ordered as (1, n_feats, frames)
        # filterbanks are ordered as (n_mels, frames)
        selected_utters = []
        if self.use_pase:
            for full_utter in sampled_utters:
                utter_feats = np.load(os.path.join(path_to_spkr_utts, full_utter))
                start_frame = np.random.randint(0, utter_feats.shape[2]-self.num_frames)
                partial_utt = utter_feats[:,:,start_frame:start_frame+self.num_frames]
                selected_utters.append(partial_utt)
            selected_utters = np.concatenate(selected_utters, axis=0)
        else:
            for full_utter in sampled_utters:
                utter_feats = np.load(os.path.join(path_to_spkr_utts, full_utter))
                start_frame = np.random.randint(0, utter_feats.shape[1]-self.num_frames)
                partial_utt = utter_feats[:,start_frame:start_frame+self.num_frames]
                selected_utters.append(partial_utt)
            selected_utters = np.stack(selected_utters, axis=0)  # (batch, n_feats, frames)

        # transpose to (batch, frames, n_feats)
        selected_utters = torch.tensor(np.transpose(selected_utters, axes=(0,2,1)))
        return selected_utters





