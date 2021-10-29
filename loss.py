import torch
from torch.nn import L1Loss, Module
import torch.nn.functional as F
import math

from utils.utils import gamma2as


def diffusion_elbo(gamma_0, gamma_1, d_gamma_t,
                   x, noise, noise_hat):
    alpha_0, var_0 = gamma2as(gamma_0)
    alpha_1, var_1 = gamma2as(gamma_1)

    # prior loss KL(q(z_1|x) || p(z_1)))
    mu = alpha_1 * x
    prior_loss = 0.5 * torch.mean(var_1 + mu * mu - 1 - var_1.log())

    # recon loss E[-log p(x | z_0)]
    # z_0 = alpha_0 * x + var_0.sqrt() * torch.randn_like(x)
    diff = (1 - alpha_0) * x - var_0.sqrt() * torch.randn_like(x)
    l2 = diff * diff
    # var = l2.mean().detach()
    var = 1e-4
    ll = -0.5 * (math.log(var) + l2 / var + math.log(2 * math.pi)).mean()
    # ll = -0.5 * (F.mse_loss(z_0, x) + math.log(2 * math.pi))
    recon_loss = -ll

    extra_dict = {
        'kld': prior_loss.item(),
        'll': ll.item()
    }
    # diffusion loss
    diff = noise - noise_hat
    loss_T_raw = 0.5 * (d_gamma_t * (diff * diff).mean(1)
                        ) / d_gamma_t.shape[0]
    loss_T = loss_T_raw.sum()
    extra_dict['loss_T_raw'] = loss_T_raw.detach()
    extra_dict['loss_T'] = loss_T.item()

    loss = prior_loss + recon_loss + loss_T
    elbo = -loss
    extra_dict['elbo'] = elbo.item()
    return loss, extra_dict
