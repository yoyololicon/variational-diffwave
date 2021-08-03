# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


@torch.jit.script
def gru(x: torch.Tensor):
    a, b = x.chunk(2, 1)
    return a.tanh() * b.sigmoid()


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(
            max_steps), persistent=False)
        self.projection1 = nn.Linear(128, 512)
        self.projection2 = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        # table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        table = torch.view_as_real(torch.exp(1j * table)).view(max_steps, -1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.ConvTranspose2d(1, 1, [3, 32], stride=[
                1, 16], padding=[1, 8]),
            nn.LeakyReLU(0.4, inplace=True),
            nn.ConvTranspose2d(1, 1,  [3, 32], stride=[
                1, 16], padding=[1, 8]),
            nn.LeakyReLU(0.4, inplace=True)
        )
        self.hop_size = 256

    def forward(self, x):
        return self.convs(x.unsqueeze(1)).squeeze(1)


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, last_layer=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.dilated_conv = nn.Conv1d(
            residual_channels + n_mels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)

        self.chs_split = [residual_channels]
        if last_layer:
            self.output_projection = nn.Conv1d(
                residual_channels, residual_channels, 1)
        else:
            self.chs_split.append(residual_channels)
            self.output_projection = nn.Conv1d(
                residual_channels, residual_channels * 2, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(
            diffusion_step).unsqueeze(-1)

        y = self.dilated_conv(torch.cat([x + diffusion_step, conditioner], 1))
        y = gru(y)
        *residual, skip = self.output_projection(y).split(self.chs_split, 1)
        return (x + residual[0]) / sqrt(2.0) if len(residual) else None, skip


class DiffWave(nn.Module):
    def __init__(self,
                 res_channels: int,
                 T: int,
                 n_mels: int,
                 layers: int,
                 cycle_length: int
                 ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv1d(1, res_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.diffusion_embedding = DiffusionEmbedding(T)
        self.spectrogram_upsampler = SpectrogramUpsampler(n_mels)

        dilations = [2 ** (i % cycle_length) for i in range(layers)]
        self.residual_layers = nn.ModuleList([
            ResidualBlock(n_mels, res_channels, d) for d in dilations
        ])

        self.output_projection = nn.Sequential(
            nn.Conv1d(res_channels, res_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(res_channels, 1, 1)
        )
        nn.init.zeros_(self.output_projection[2].weight)

    def forward(self, audio, spectrogram, diffusion_step):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = 0
        for layer in self.residual_layers:
            x, skip_connection = layer(x, spectrogram, diffusion_step)
            skip += skip_connection

        x = skip / sqrt(len(self.residual_layers))
        x = self.output_projection(x)
        return x
