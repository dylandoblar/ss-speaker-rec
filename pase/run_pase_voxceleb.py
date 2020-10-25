import sys
import os
import torch
import librosa

from pase.models.frontend import wf_builder

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_pase_representations(pase_model, audio_path):
    y, fs = librosa.core.load(audio_path, sr=None)
    y = torch.tensor(y)[(None,)*2].to(device)  # unsqueeze twice at first dim
    pase_reps = pase_model(y)
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
    for speaker_id in speaker_ids:
        if speaker_id.startswith('.'):
            speaker_ids.remove(speaker_id)

    # get PASE representations for each speaker
    for i, speaker_id in enumerate(speaker_ids):
        if i % 100 == 0:
            print(f"processing {i}th speaker...")

        if os.path.exists(os.path.join(save_path, speaker_id)):
            continue
        
        os.makedirs(os.path.join(save_path, speaker_id), exist_ok=True)
        path_to_speaker = os.path.join(audio_path, speaker_id)
        
        video_ids = os.listdir(path_to_speaker)
        for video_id in video_ids:
            if video_id.startswith('.'):
                video_ids.remove(video_id)
                continue
        
            path_to_utters = os.path.join(path_to_speaker, video_id)
            utterances = os.listdir(path_to_utters)
            for j, utt in enumerate(utterances):
                if int(utt[:-4]) < 50:
                    utter_path = os.path.join(path_to_utters, utt)
                    pase_reps = get_pase_representations(pase, utter_path)
                     
                    save_filename = utt[:-4]+'.pt'
                    save_file_path = os.path.join(save_path, speaker_id, save_filename)
                    torch.save(pase_reps, save_file_path)




