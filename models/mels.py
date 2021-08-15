from torchaudio import transforms
from torch import Tensor
from typing import Optional, Callable
import torch.nn.functional as F


class MelSpec(transforms.MelSpectrogram):
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, f_min: float, f_max: float, n_mels: int) -> None:
        super().__init__(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, f_min=f_min, f_max=f_max, n_mels=n_mels,
                         power=1., normalized=True)

    def forward(self, waveform: Tensor) -> Tensor:
        waveform = F.pad(waveform, [0, self.n_fft // 2])
        mels = super().forward(waveform)
        mels.clamp_min_(
            1e-5).log10_().mul_(20).add_(80).mul_(0.01).clamp_(0, 1)
        return mels
