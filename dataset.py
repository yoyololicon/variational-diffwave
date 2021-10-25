import os
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import torchaudio


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

        print("Gathering training files ...")
        for f in tqdm(os.listdir(self.data_path)):
            if f.endswith('.wav'):
                filename = os.path.join(self.data_path, f)
                meta = torchaudio.info(filename)
                self.files.append(filename)
                file_lengths.append(max(1, meta.num_frames - segment + 1))

                if not self.sr:
                    self.sr = meta.sample_rate
                else:
                    assert meta.sample_rate == self.sr

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
        x = torchaudio.load(f, frame_offset=pos,
                            num_frames=self.segment)[0].mean(0)

        return x
