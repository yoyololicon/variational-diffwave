import torch
from torch import nn
from torch.autograd import grad
import torch.nn.utils.parametrize as parametrize


class Nonnegative(nn.Module):
    def forward(self, X):
        return X.abs()


class NoiseScheduler(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = parametrize.register_parametrization(
            nn.Linear(1, 1, bias=True), 'weight', Nonnegative())
        self.l2 = parametrize.register_parametrization(
            nn.Linear(1, 1024, bias=True), 'weight', Nonnegative())
        self.l3 = parametrize.register_parametrization(
            nn.Linear(1024, 1, bias=False), 'weight', Nonnegative())

        self.gamma1 = nn.Parameter(torch.ones(1) * 5, requires_grad=True)
        self.gamma0 = nn.Parameter(torch.ones(1) * -8, requires_grad=True)

    def gamma_hat(self, t: torch.Tensor):
        l1 = self.l1(t)
        return l1 + self.l3(self.l2(t).sigmoid())

    def forward(self, t: torch.Tensor):
        t = t.clamp(0, 1).unsqueeze(-1)
        max_gamma_hat = self.gamma_hat(torch.ones_like(t))
        min_gamma_hat = self.gamma_hat(torch.zeros_like(t))
        gamma_hat = self.gamma_hat(t)
        gamma0, gamma1 = self.gamma0, self.gamma1
        normalized_gamma_hat = (gamma_hat - min_gamma_hat) / \
            (max_gamma_hat - min_gamma_hat)
        # gamma = gamma_hat
        gamma = gamma0 + (gamma1 - gamma0) * normalized_gamma_hat

        return gamma.squeeze(1), normalized_gamma_hat.squeeze(1)


if __name__ == '__main__':
    gamma = NoiseScheduler()

    t = torch.arange(10) / 9
    t = torch.tensor(t,  requires_grad=True)
    g = gamma(t)
    print(t, g)

    print(grad(g.sum(), t, only_inputs=True))
    # g.sum().backward()
