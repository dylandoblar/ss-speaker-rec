#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import librosa
import numpy as np
from tqdm import tqdm

from hparam import hparam as hp


def build_log_mel_spectrum(y, fs, n_fft, window, hop, nmels=40):
    D = librosa.core.stft(y=y, n_fft=n_fft,
                          win_length=int(window*fs), hop_length=int(hop*fs))
    D = np.abs(D) ** 2
    mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=nmels)

    # log mel spectrogram of utterances
    S = np.log10(np.dot(mel_basis, D) + 1e-6)
    return S

def preprocess(audio_path, save_path):
    # get list of speaker ids from VoxCeleb
    speaker_ids = os.listdir(audio_path)
    for speaker_id in speaker_ids:
        if speaker_id.startswith('.'):
            speaker_ids.remove(speaker_id)

    # get log mel filterbanks for utterances from each speaker
    for speaker_id in tqdm(speaker_ids):
        os.makedirs(os.path.join(save_path, speaker_id), exist_ok=True)
        path_to_speaker = os.path.join(audio_path, speaker_id)

        video_ids = os.listdir(path_to_speaker)
        utt_idx = 1
        for video_id in sorted(video_ids):
            path_to_utters = os.path.join(path_to_speaker, video_id)
            utterances = os.listdir(path_to_utters)
            for utt in sorted(utterances):
                save_filename = "%05d" % utt_idx + '.npy'
                save_file_path = os.path.join(save_path, speaker_id, save_filename)

                utter_path = os.path.join(path_to_utters, utt)
                y, fs = librosa.core.load(utter_path, sr=None)
                S = build_log_mel_spectrum(y, fs, hp.data.nfft, hp.data.window, hp.data.hop)
                np.save(save_file_path, S)

                utt_idx += 1


if __name__ == "__main__":
    audio_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocess(audio_path, save_path)



