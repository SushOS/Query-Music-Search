import os
import pandas as pd
from torch.utils.data import Dataset


def get_index(address):
    i = len(address) - 1
    index = ""
    while i >= 0 and address[i] != "-":
        index = address[i] + index
        i -= 1
    try:
        ind = int(index)
        return ind
    except:
        return None


class AudioDataset(Dataset):
    def __init__(self, audio_folder_path, captions_csv_path):
        super().__init__()
        raw_audio_paths = [
            os.path.join(audio_folder_path, file)
            for file in os.listdir(audio_folder_path)
        ]
        captions = pd.read_csv(captions_csv_path)["caption"]
        self.audio_paths = []
        for audio in raw_audio_paths:
            idx = get_index(audio)
            if idx is not None:
                self.audio_paths.append(audio)
        self.captions = captions

    def __len__(self):
        return min(len(self.audio_paths), len(self.captions))

    def __getitem__(self, index):
        ind_val = get_index(self.audio_paths[index])
        return (self.audio_paths[index], self.captions[ind_val])
