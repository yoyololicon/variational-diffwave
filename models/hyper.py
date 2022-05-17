import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DiffusionEmbedding, gru


class ConvWeightPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 hidden_channels=64,
                 layers=3):
        super().__init__()

        self.groups = groups

        self.start = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels * groups, 1),
            nn.LeakyReLU(0.2)
        )

        self.end = nn.Conv1d(hidden_channels * groups,
                             out_channels * groups, 1, groups=groups)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels * groups, hidden_channels *
                          groups, 1, groups=groups),
                nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_channels * groups, hidden_channels *
                          groups, 1, groups=groups),
                nn.LeakyReLU(0.2)
            ) for _ in range(layers)
        ])

    def forward(self, x):
        x = self.start(x)
        for block in self.res_blocks:
            x = block(x) + x
        return self.end(x)


class HyperDiffWave(nn.Module):
    def __init__(self,
                 res_channels: int = 64,
                 T: int = 1,
                 n_emb: int = 256,
                 layers: int = 40,
                 cycle_length: int = 8,
                 dilation_base: int = 3
                 ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Conv1d(1, res_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.diffusion_embedding = DiffusionEmbedding(T)
        self.output_projection = nn.Sequential(
            nn.Conv1d(res_channels, res_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(res_channels, 1, 1)
        )

        self.dilations = [dilation_base ** (i % cycle_length)
                          for i in range(layers)]
        self.res_channels = res_channels

        self.noise_weight_predictor = ConvWeightPredictor(
            512, 2 * 3 * res_channels ** 2, groups=layers, layers=1,
        )

        self.speaker_weight_predictor = ConvWeightPredictor(
            n_emb, 2 * res_channels ** 2, groups=layers - 1, layers=1
        )
        self.final_speaker_weight_predictor = ConvWeightPredictor(
            n_emb, res_channels ** 2, groups=1, layers=1
        )

    def forward(self, audio, diffusion_step, speaker_emb):
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        N = x.size(0)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_weights = self.noise_weight_predictor(
            diffusion_step.unsqueeze(2)).squeeze(2).chunk(len(self.dilations), 1)
        speaker_weights = self.speaker_weight_predictor(
            speaker_emb.unsqueeze(2)).squeeze(2).chunk(len(self.dilations) - 1, 1)
        speaker_weights += (self.final_speaker_weight_predictor(
            speaker_emb.unsqueeze(2)).squeeze(2), )

        skip = 0
        x = x.view(1, -1, x.size(2))
        for i, (dw, sw, d) in enumerate(zip(diffusion_weights, speaker_weights, self.dilations)):
            dw = dw.reshape(N * self.res_channels * 2, self.res_channels, 3)
            y = F.conv1d(x, dw, padding=d, dilation=d, groups=N)
            y = gru(y.view(N, -1, y.size(2))).view(1, -1, y.size(2))

            sw = sw.reshape(-1, self.res_channels, 1)
            y = F.conv1d(y, sw, groups=N)
            if i < len(self.dilations) - 1:
                res, skip_connection = y.view(N, -1, y.size(2)).chunk(2, 1)
                x = (x + res.reshape(1, -1, res.size(2))) / math.sqrt(2)
            else:
                skip_connection = y
            skip += skip_connection.reshape(1, -1, skip_connection.size(2))

        x = skip.view(N, -1, skip.size(2)) / math.sqrt(len(self.dilations))
        x = self.output_projection(x).squeeze(1)
        return x
