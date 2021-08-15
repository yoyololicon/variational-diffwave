from torch.autograd import grad
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.cuda import amp
from torchinfo import summary
import argparse
import json
from datetime import datetime
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from random import randrange, sample, uniform
from jsonschema import validate
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import ModelCheckpoint, TerminateOnNan, Checkpoint, EMAHandler
from ignite.contrib.handlers.tensorboard_logger import *
import torch_optimizer
# from pixyz.losses import KullbackLeibler, LogProb
# from pixyz.distributions import Normal
import soundfile as sf

from utils.schema import CONFIG_SCHEMA
from utils.utils import get_instance, gamma2snr, snr2as, gamma2as
import models as module_arch
import dataset as module_data
import loss as module_loss


parser = argparse.ArgumentParser(description='DiffWave')
parser.add_argument('config', type=str, help='config file')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='training checkpoint')
parser.add_argument('-d', '--device', default=0,
                    type=int, help='cuda device number')
parser.add_argument('--device_ids',  default=None, type=int, nargs='+',
                    help='indices of GPUs for DataParallel (default: None)')

args = parser.parse_args()

config = json.load(open(args.config))
validate(config, schema=CONFIG_SCHEMA)

if torch.cuda.is_available():
    device = f"cuda:{args.device}"
    device_ids = args.device_ids
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'


train_data = get_instance(module_data, config['dataset'])
train_loader = DataLoader(train_data, **config['data_loader'])
model = get_instance(module_arch, config['arch']).to(device)

if device_ids:
    print(f'using multi-GPU')
    model = nn.DataParallel(model, device_ids=device_ids)

noise_scheduler = module_arch.NoiseScheduler().to(device)

parameters = chain(model.parameters(), noise_scheduler.parameters())

try:
    optimizer = get_instance(optim, config['optimizer'], parameters)
except AttributeError:
    optimizer = get_instance(
        torch_optimizer, config['optimizer'], parameters)

scheduler = get_instance(optim.lr_scheduler, config['lr_scheduler'], optimizer)
criterion = module_loss.diffusion_elbo


model_name = config['name']
extra_monitor = config['trainer']['extra_monitor']
log_dir = config['trainer']['log_dir']
save_dir = config['trainer']['save_dir']
eval_file = config['trainer']['eval_file']
n_fft = config['trainer']['n_fft']
hop_length = config['trainer']['hop_length']
n_mels = config['trainer']['n_mels']
sr = config['trainer']['sr']
eval_interval = config['trainer']['eval_interval']
train_T = config['trainer']['train_T']
eval_T = config['trainer']['eval_T']


# T = 50
# beta = torch.linspace(1e-4, 0.05, T).to(device)
# noise_level = torch.cumprod(1 - beta, 0)

# infer_beta = torch.tensor([0.0001, 0.001, 0.01, 0.05, 0.2, 0.5], device=device)

mel_spec = module_arch.MelSpec(sr, n_fft, hop_length=hop_length,
                               f_min=20, f_max=8000, n_mels=n_mels).to(device)


eval_x, eval_sr = sf.read(os.path.expanduser(
    eval_file), always_2d=True, dtype='float32')
assert sr == eval_sr
eval_x = eval_x.mean(1)
eval_x = torch.from_numpy(eval_x).to(device).unsqueeze(0)
eval_mels = mel_spec(eval_x)


def _discrete_proc(x, mels):
    N = x.shape[0]
    s = torch.remainder(uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
    s_idx = torch.round(s * (train_T - 1)).long()
    t_idx = s_idx + 1

    t, s = t_idx / train_T, s_idx / train_T
    gamma_t = noise_scheduler(t)
    gamma_s = noise_scheduler(s)
    alpha_t, var_t = gamma2as(gamma_t)
    noise = torch.randn_like(x)
    z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

    noise_hat = model(z_t, mels, t_idx)

    loss, extra_dict = criterion(
        noise_scheduler.gamma0,
        noise_scheduler.gamma1,
        (gamma_t - gamma_s) * train_T,
        x, noise, noise_hat)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    result = {'loss': loss.item()}
    result.update(extra_dict)
    return result


def _contiguous_proc(x, mels):
    N = x.shape[0]
    t = torch.remainder(uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
    t = t.clone().detach().requires_grad_(True)

    gamma_t = noise_scheduler(t)

    alpha_t, var_t = gamma2as(gamma_t)
    noise = torch.randn_like(x)
    z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

    noise_hat = model(z_t, mels, t.detach())

    d_gamma_t, *_ = grad(gamma_t.sum(), t, only_inputs=True, create_graph=True)
    loss, extra_dict = criterion(
        noise_scheduler.gamma0,
        noise_scheduler.gamma1,
        d_gamma_t,
        x, noise, noise_hat)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    result = {'loss': loss.item()}
    result.update(extra_dict)
    return result


def process_function(engine, batch):
    model.train()
    noise_scheduler.train()

    x = batch
    x = x.to(device)

    mels = mel_spec(x)
    if train_T:
        return _discrete_proc(x, mels)
    return _contiguous_proc(x, mels)


trainer = Engine(process_function)

RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
for k in extra_monitor:
    RunningAverage(output_transform=lambda x, m=k: x[m]).attach(trainer, k)


# Tqdm
pbar = ProgressBar(persist=True)
pbar.attach(trainer, 'all')

# Create a logger
start_time = datetime.now().strftime('%m%d_%H%M%S')
tb_logger = TensorboardLogger(
    log_dir=os.path.join(log_dir, model_name, start_time))
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    metric_names='all'
)
tb_logger.attach(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=eval_interval),
    log_handler=WeightsHistHandler(model)
)
tb_logger.attach(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=eval_interval),
    log_handler=WeightsHistHandler(noise_scheduler)
)
# tb_logger.attach_opt_params_handler(
#     trainer,
#     event_name=Events.ITERATION_STARTED,
#     optimizer=optimizer,
# )

# add model graph
# use torchinfo
test_input = torch.from_numpy(train_data[0]).to(device).unsqueeze(0)
test_mels = mel_spec(test_input)
t = torch.tensor([0], device=device)

summary(model,
        input_data=(test_input, test_mels, t),
        device=device,
        col_names=("input_size", "output_size", "num_params", "kernel_size",
                   "mult_adds"),
        col_width=16,
        row_settings=("depth", "var_names"))


ema_handler = EMAHandler(model, momentum=0.0001)
ema_model = ema_handler.ema_model

checkpointer = ModelCheckpoint(
    save_dir, model_name, n_saved=2, create_dir=True, require_empty=False)
to_save = {
    'model': model,
    'ema_model': ema_model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'trainer': trainer,
    'noise_scheduler': noise_scheduler
}
trainer.add_event_handler(
    Events.ITERATION_COMPLETED(every=eval_interval),
    checkpointer,
    to_save
)


@torch.no_grad()
def predict_samples(engine):
    # model.eval()
    noise_scheduler.eval()

    # z_t = torch.randn(
    #     1, hop_length * (eval_mels.shape[-1] - 1) + 1, device=device)
    z_t = torch.randn_like(eval_x)
    if train_T:
        steps = torch.linspace(0, train_T, eval_T + 1,
                               device=device).round().long()
        gamma = noise_scheduler(steps / train_T)
    else:
        steps = torch.linspace(0, 1, eval_T + 1, device=device)
        gamma = noise_scheduler(steps)

    alpha, var = gamma2as(gamma)
    var_ts = -torch.expm1(F.softplus(gamma[:-1]) - F.softplus(gamma[1:]))
    print(alpha, var, var_ts)

    kld = 0
    for t in tqdm(range(eval_T, 0, -1)):
        s = t - 1
        noise_hat = ema_model(z_t, eval_mels, steps[t:t+1])
        alpha_ts = alpha[t] / alpha[s]

        noise = (z_t - alpha[t] * eval_x) * var[t].rsqrt()
        kld += 0.5 * (gamma[t] - gamma[s]) * F.mse_loss(noise_hat, noise)

        mu = (z_t - var_ts[s] * var[t].rsqrt() * noise_hat) / alpha_ts
        z_t = mu
        if s:
            z_t += (var_ts[s] * var[s] / var[t]).sqrt() * torch.rand_like(z_t)

    ll = -0.5 * (F.mse_loss(z_t, eval_x).log() +
                 1 + math.log(2 * math.pi)) - kld

    print("Log likelihood:", ll.item())

    predict = z_t.squeeze().clip(-0.99, 0.99)
    tb_logger.writer.add_audio(
        'predict', predict, engine.state.iteration, sample_rate=sr)


trainer.add_event_handler(Events.ITERATION_COMPLETED(
    every=eval_interval), predict_samples)


@torch.no_grad()
def plot_noise_curve(engine):
    figure = plt.figure()
    steps = torch.linspace(0, 1, 100, device=device)
    log_snr = -noise_scheduler(steps).detach().cpu().numpy()
    steps = steps.cpu().numpy()
    plt.plot(steps, log_snr)
    tb_logger.writer.add_figure(
        'log_snr', figure, engine.state.iteration)


trainer.add_event_handler(Events.ITERATION_COMPLETED(
    every=eval_interval), plot_noise_curve)

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'ema_model' not in checkpoint:
        checkpoint['ema_model'] = checkpoint['model']
    Checkpoint.load_objects(
        to_load=to_save, checkpoint=checkpoint, strict=False)


ema_handler.attach(trainer, name="ema_momentum",
                   event=Events.ITERATION_COMPLETED)

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

e = trainer.run(train_loader, max_epochs=1)

tb_logger.close()
