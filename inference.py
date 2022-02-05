import torch
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm
from utils.utils import gamma2snr, snr2as, gamma2as, gamma2logas


def reverse_process(z_1, mels, gamma, steps, model, with_amp=False):
    assert gamma.size() == steps.size()
    snr = gamma2snr(gamma)
    alpha, var = snr2as(snr)
    var_ts = - \
        torch.expm1(F.softplus(gamma[:-1]) - F.softplus(gamma[1:]))
    var_ts.relu_()

    T = gamma.numel() - 1
    z_t = z_1
    for t in tqdm(range(T, 0, -1)):
        s = t - 1
        with amp.autocast(enabled=with_amp):
            noise_hat = model(z_t, mels, steps[t:t+1])
        noise_hat = noise_hat.float()
        alpha_ts = alpha[t] / alpha[s]

        # noise = (z_t - alpha[t] * eval_x) * var[t].rsqrt()
        # kld += 0.5 * torch.expm1(gamma[t] - gamma[s]) * \
        # F.mse_loss(noise_hat, noise)
        # kld += 0.5 * (gamma[t] - gamma[s]) * \
        #     F.mse_loss(noise_hat, noise)

        mu = (z_t - var_ts[s] * var[t].rsqrt()
              * noise_hat) / alpha_ts
        z_t = mu
        if s:
            z_t += (var_ts[s] * var[s] / var[t]).sqrt() * \
                torch.randn_like(z_t)

    return z_t


def reverse_process_new(z_1, mels, gamma, steps, model, with_amp=False):
    log_alpha, log_var = gamma2logas(gamma)
    var = log_var.exp()
    alpha_st = torch.exp(log_alpha[:-1] - log_alpha[1:])
    c = -torch.expm1(gamma[:-1] - gamma[1:])
    c.relu_()

    T = gamma.numel() - 1
    z_t = z_1
    for t in tqdm(range(T, 0, -1)):
        s = t - 1
        with amp.autocast(enabled=with_amp):
            noise_hat = model(z_t, mels, steps[t:t+1])
        noise_hat = noise_hat.float()
        mu = (z_t - var[t].sqrt() * c[s] * noise_hat) * alpha_st[s]
        z_t = mu
        if s:
            z_t += (var[s] * c[s]).sqrt() * torch.randn_like(z_t)

    return z_t
