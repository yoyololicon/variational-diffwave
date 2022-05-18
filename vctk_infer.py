import argparse
from copy import deepcopy
import os
from pathlib import Path
import hydra
import numpy as np
from omegaconf import OmegaConf
import torchaudio
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.cuda import amp
from tqdm import tqdm
from typing import Callable, List, Tuple
from multiprocessing import Process, Queue, set_start_method
from functools import partial
from kazane import Decimate, Upsample

from utils.utils import gamma2snr, snr2as, gamma2as, gamma2logas, get_instance
import models as module_arch

SAMPLERATE = 48000


class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    # x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        origin_shape = x.shape
        T = origin_shape[-1]
        x = x.view(-1, T)

        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )  # return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2], 1, 1)
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )  # return_complex=False)
        x = x[:, :T].detach()
        return x.view(*origin_shape)


class STFTDecimate(LowPass):
    def __init__(self, r, *args, **kwargs):
        super().__init__(*args, ratio=[1 / r], **kwargs)
        self.r = r

    def forward(self, x):
        return super().forward(x, 0)[..., ::self.r]


class LSD(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, y_hat, y):
        Y_hat = torch.stft(y_hat, self.n_fft, hop_length=self.hop_length,
                           window=self.window, return_complex=True)
        Y = torch.stft(y, self.n_fft, hop_length=self.hop_length,
                       window=self.window, return_complex=True)
        sp = Y_hat.abs().square_().clamp_(min=1e-8).log10_()
        st = Y.abs().square_().clamp_(min=1e-8).log10_()
        return (sp - st).square_().mean(0).sqrt_().mean()


@torch.no_grad()
def reverse(y_hat,
            gamma,
            donwample: Callable,
            upsample: Callable,
            inference_func: Callable,
            verbose=True):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha = log_alpha.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    var_st = torch.exp(log_var[:-1] - log_var[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()
    T = gamma.numel()

    def degradation_func(x): return upsample(donwample(x))

    noise = torch.randn_like(y_hat)
    lowpass_noise = degradation_func(noise)
    z_t = y_hat * alpha[-1] + lowpass_noise * \
        var[-1].sqrt() + (noise - lowpass_noise)

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = inference_func(z_t, t)
        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]

        mu = mu - degradation_func(mu)
        mu += degradation_func(z_t) * \
            var_st[s] / alpha_st[s] + alpha[s] * c[s] * y_hat

        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    noise_hat = inference_func(z_t, 0)
    final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
    return final


@torch.no_grad()
def nuwave_reverse(y_hat, gamma, inference_model: Callable, verbose=True):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha = log_alpha.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()
    T = gamma.numel()

    z_t = torch.randn_like(y_hat)

    for t in tqdm(range(T - 1, 0, -1), disable=not verbose):
        s = t - 1
        noise_hat = inference_model(z_t, y_hat, alpha[t:t+1])
        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]

        z_t = mu
        z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    noise_hat = inference_model(z_t, y_hat, alpha[:1])
    final = (z_t - var[0].sqrt() * noise_hat) / alpha[0]
    return final


def foo(fq: Queue, rq: Queue, q: int, infer_type: str,
        model, evaluater, downsampler, upsampler, gamma, steps):
    try:
        alpha = gamma2as(gamma)[0]
        while not fq.empty():
            filename = fq.get()
            device = gamma.device

            raw_y, sr = torchaudio.load(filename)
            raw_y = raw_y.to(device)
            speaker_emb = torch.load(str(filename).replace(".wav", "_emb.pt"))
            speaker_emb = speaker_emb.to(device).unsqueeze(0)

            offset = raw_y.shape[1] % q
            if offset:
                y = raw_y[:, :-offset]
            else:
                y = raw_y

            y_lowpass = downsampler(y)

            if infer_type == "nuwave":
                y_hat = F.upsample(y_lowpass.unsqueeze(
                    1), scale_factor=q, mode='linear', align_corners=False).squeeze(1)
                y_recon = nuwave_reverse(y_hat, gamma,
                                         amp.autocast()(model),
                                         verbose=False)
            elif infer_type == "inpainting":
                y_hat = upsampler(y_lowpass)
                y_recon = reverse(
                    y_hat, gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast()(lambda x, t: model(
                        x, steps[t:t+1], speaker_emb)),
                    verbose=False
                )
            elif infer_type == "nuwave-inpainting":
                y_hat = upsampler(y_lowpass)
                nuwave_cond = F.upsample(y_lowpass.unsqueeze(
                    1), scale_factor=q, mode='linear', align_corners=False).squeeze(1)
                y_recon = reverse(
                    y_hat, gamma,
                    amp.autocast()(downsampler),
                    amp.autocast()(upsampler),
                    amp.autocast()(lambda x, t: model(
                        x, nuwave_cond, alpha[t:t+1])),
                    verbose=False
                )
            else:
                raise ValueError(
                    "infer_type must be one of nuwave, inpainting, nuwave-inpainting")

            if offset:
                y_recon = torch.cat(
                    [y_recon, y_recon.new_zeros(1, offset)], dim=1)

            lsd = evaluater(y_recon.squeeze(), raw_y.squeeze()).item()
            rq.put((filename, lsd))

    except Exception as e:
        rq.put((filename, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('cfg', type=str)
    parser.add_argument('vctk', type=str)
    parser.add_argument('--log-snr', type=str)
    parser.add_argument('--nuwave-ckpt', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--rate', type=int, default=2)
    parser.add_argument('-T', type=int, default=50)
    parser.add_argument('--infer-type', type=str,
                        choices=['inpainting', 'nuwave', 'nuwave-inpainting'], default='inpainting')
    parser.add_argument('--downsample-type', type=str,
                        choices=['sinc', 'stft'], default='stft')

    args = parser.parse_args()

    set_start_method('spawn')

    gpus = torch.cuda.device_count()

    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    cfg = OmegaConf.load(args.cfg)
    if 'nuwave' in args.infer_type:
        model = module_arch.NuWave()
        state_dict = torch.load(
            args.nuwave_ckpt, map_location=torch.device('cpu'))
        state_dict = dict((x[6:], y)
                          for x, y in state_dict.items() if x.startswith('model.'))
        model.load_state_dict(state_dict)
    else:
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(checkpoint['ema_model'])
    model.eval()

    if cfg.train_T > 0:
        scheduler = module_arch.NoiseScheduler()
    else:
        scheduler = module_arch.LogSNRLinearScheduler()
    scheduler.load_state_dict(checkpoint['noise_scheduler'])
    scheduler.eval()
    scheduler = scheduler.cuda()

    if args.log_snr:
        gamma0, gamma1 = scheduler.gamma0.detach().cpu(
        ).numpy(), scheduler.gamma1.detach().cpu().numpy()
        log_snr = np.loadtxt(args.log_snr)
        xp = np.arange(len(log_snr))
        x = np.linspace(xp[0], xp[-1], args.T)
        gamma = -np.interp(x, xp, log_snr)
        steps = (gamma - gamma0) / (gamma1 - gamma0)
        gamma, steps = torch.tensor(gamma, dtype=torch.float32), torch.tensor(
            steps, dtype=torch.float32)
    else:
        t = torch.linspace(0, 1, args.T + 1).cuda()
        with torch.no_grad():
            gamma, steps = scheduler(t)

    sinc_kwargs = {
        'q': args.rate,
        'roll_off': 0.962,
        'num_zeros': 128,
        'window_func': partial(torch.kaiser_window, periodic=False,
                               beta=14.769656459379492),
    }

    file_q = Queue()
    result_q = Queue()
    processes = []

    for i in range(gpus):
        device = f'cuda:{i}'
        evaluater = LSD()
        if args.downsample_type == 'sinc':
            downsampler = Decimate(**sinc_kwargs)
        else:
            downsampler = STFTDecimate(sinc_kwargs['q'])
        upsampler = Upsample(**sinc_kwargs)

        p = Process(target=foo, args=(
            file_q, result_q, args.rate, args.infer_type,
            deepcopy(model).to(device), evaluater.to(device), downsampler.to(
                device), upsampler.to(device), gamma.to(device), steps.to(device)))
        processes.append(p)

    vctk_path = Path(args.vctk)
    test_files = list(vctk_path.glob('*/*.wav'))

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    for filename in test_files:
        file_q.put(filename)

    for p in processes:
        p.start()

    pbar = tqdm(total=len(test_files))
    n = 0

    lsd_list = []
    try:
        while n < len(test_files):
            filename, lsd = result_q.get()
            if isinstance(lsd, Exception):
                print(f'catch exception: {lsd}')
                break
            pbar.set_postfix(lsd=lsd)
            pbar.update(1)
            n += 1
            lsd_list.append(lsd)
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        for p in processes:
            p.join()

    print(sum(lsd_list) / len(lsd_list))

    # pbar = tqdm(test_files)
    # for filename in pbar:
    #     raw_y, sr = torchaudio.load(filename)
    #     raw_y = raw_y.cuda()
    #     speaker_emb = torch.load(str(filename).replace(".wav", "_emb.pt"))
    #     speaker_emb = speaker_emb.cuda().view(1, -1)

    #     offset = raw_y.shape[1] % args.rate
    #     if offset:
    #         y = raw_y[:, :-offset]
    #     else:
    #         y = raw_y

    #     y_hat = upsampler(downsampler(y))

    #     y_recon = reverse(
    #         y_hat, gamma,
    #         amp.autocast()(downsampler),
    #         amp.autocast()(upsampler),
    #         amp.autocast()(lambda x, t: model(x, steps[t:t+1], speaker_emb)),
    #         verbose=False
    #     )

    #     if offset:
    #         y_recon = torch.cat([y_recon, y_recon.new_zeros(1, offset)], dim=1)

    #     lsd = evaluater(y_recon.squeeze(), raw_y.squeeze()).item()
    #     pbar.set_postfix(lsd=lsd)

    #     if args.out_dir:
    #         sub_folders = str(filename.parent).replace(
    #             str(vctk_path), '').split('/')
    #         out_filename = PurePath(args.out_dir, *sub_folders, filename.name)
    #         os.makedirs(out_filename.parent, exist_ok=True)
    #         torchaudio.save(out_filename, y_recon.cpu(), sr)
