import sys
import os
import torch
import librosa
import numpy as np
from tqdm import tqdm

from pase.models.frontend import wf_builder

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_pase_representations(pase_model, audio_path):
    y, fs = librosa.core.load(audio_path, sr=None)
    y = torch.tensor(y)[(None,)*2].to(device)  # unsqueeze twice at first dim
    pase_reps = pase_model(y)
    pase_reps = pase_reps.detach().cpu().numpy()
    return pase_reps

if __name__ == "__main__":
    audio_path = sys.argv[1]
    save_path = sys.argv[2]
    
    # load model
    pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
    pase = pase.to(device)
    pase.load_pretrained('checkpoints/pase_pretrained.ckpt', load_last=True, verbose=True)

    # get list of speaker ids from VoxCeleb
    speaker_ids = os.listdir(audio_path)

    # get PASE representations for utterances from each speaker
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
                pase_reps = get_pase_representations(pase, utter_path)                
                np.save(save_file_path, pase_reps)
                
                utt_idx += 1



