import os
import numpy as np
import soundfile as sf
import random
from torch.utils.data import Dataset
from tqdm import tqdm


class RandomWAVDataset(Dataset):
    """
    Wave-file-only Dataset.
    """

    def __init__(self,
                 data_dir,
                 size,
                 segment):
        self.segment = segment
        self.data_path = os.path.expanduser(data_dir)
        self.size = size

        self.waves = []
        self.sr = None
        self.files = []

        file_lengths = []

        def get_nframes(info_str):
            try:
                return int(f_obj.extra_info.split('frames  : ')[1].split('\n')[0])
            except:
                byte_sec = int(info_str.split(
                    'Bytes/sec     : ')[1].split('\n')[0])
                data = int(info_str.split('data : ')[1].split('\n')[0])
                sr = int(info_str.split('Sample Rate   : ')[1].split('\n')[0])
                return int(data / byte_sec * sr)

        print("Gathering training files ...")
        for f in tqdm(os.listdir(self.data_path)):
            if f.endswith('.wav'):
                filename = os.path.join(self.data_path, f)
                f_obj = sf.SoundFile(filename)
                self.files.append(filename)
                file_lengths.append(
                    max(1, get_nframes(f_obj.extra_info) - segment + 1))

                if not self.sr:
                    self.sr = f_obj.samplerate
                else:
                    assert f_obj.samplerate == self.sr
                f_obj.close()
        self.file_lengths = np.array(file_lengths)
        self.boundaries = np.cumsum(
            np.array([0] + file_lengths)) / self.file_lengths.sum()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        rand_pos = random.uniform(0, 1)
        index = np.digitize(rand_pos, self.boundaries[1:], right=True)
        f, length = self.files[index], self.file_lengths[index]
        pos = int(round(length * (rand_pos - self.boundaries[index]) / (
            self.boundaries[index+1] - self.boundaries[index])))
        x = sf.read(f, self.segment, start=pos, dtype='float32',
                    always_2d=True, fill_value=0.)[0].mean(1)
        return x
