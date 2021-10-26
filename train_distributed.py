from torch.autograd import grad
import math
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.cuda import amp
import torchaudio
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
from ignite.handlers import Checkpoint, EMAHandler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.engines import common
from ignite import distributed as idist
import torch_optimizer
# from pixyz.losses import KullbackLeibler, LogProb
# from pixyz.distributions import Normal

from utils.schema import CONFIG_SCHEMA
from utils.utils import get_instance, gamma2snr, snr2as, gamma2as
import models as module_arch
import dataset as module_data
import loss as module_loss


def get_dataflow(config: dict):
    train_data = get_instance(module_data, config['dataset'])
    train_loader = idist.auto_dataloader(train_data, **config['data_loader'])
    return train_loader


def initialize(config: dict, device):
    model = get_instance(module_arch, config['arch'])
    model = idist.auto_model(model)
    noise_scheduler = module_arch.NoiseScheduler()  # .to(device)
    noise_scheduler = idist.auto_model(noise_scheduler)

    parameters = chain(model.parameters(), noise_scheduler.parameters())
    try:
        optimizer = get_instance(optim, config['optimizer'], parameters)
    except AttributeError:
        optimizer = get_instance(
            torch_optimizer, config['optimizer'], parameters)
    optimizer = idist.auto_optim(optimizer)

    scheduler = get_instance(
        optim.lr_scheduler, config['lr_scheduler'], optimizer)

    return model, noise_scheduler, optimizer, scheduler


def get_logger(trainer, model, noise_scheduler, optimizer, log_dir, model_name, interval):
    # Create a logger
    start_time = datetime.now().strftime('%m%d_%H%M%S')
    tb_logger = common.setup_tb_logging(
        output_path=os.path.join(log_dir, model_name, start_time),
        trainer=trainer,
        optimizers=optimizer,
        log_every_iters=1
    )

    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=interval),
        log_handler=WeightsHistHandler(model)
    )
    tb_logger.attach(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=interval),
        log_handler=WeightsHistHandler(noise_scheduler)
    )

    return tb_logger


def create_trainer(model, mel_spec, noise_scheduler, optimizer, criterion, scheduler, device, trainer_config, train_sampler, model_name: str, checkpoint_path: str):
    extra_monitor = trainer_config['extra_monitor']
    save_dir = trainer_config['save_dir']
    eval_interval = trainer_config['eval_interval']
    train_T = trainer_config['train_T']
    minimize_var = trainer_config['minimize_var']
    with_amp = trainer_config['with_amp']

    rank = idist.get_rank()

    scaler = amp.GradScaler(enabled=with_amp)

    if isinstance(noise_scheduler, nn.parallel.DistributedDataParallel) or isinstance(noise_scheduler, nn.parallel.DataParallel):
        base_noise_scheduler = noise_scheduler.module
    else:
        base_noise_scheduler = noise_scheduler

    def process_function(engine, batch):
        model.train()
        noise_scheduler.train()
        optimizer.zero_grad()

        x = batch
        x = x.to(device)
        noise = torch.randn_like(x)
        mels = mel_spec(x)
        with amp.autocast(enabled=with_amp):
            N = x.shape[0]
            if train_T:
                s = torch.remainder(
                    uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
                s_idx = torch.round(s * (train_T - 1)).long()
                t_idx = s_idx + 1

                t, s = t_idx / train_T, s_idx / train_T
                gamma_t = noise_scheduler(t)
                gamma_s = noise_scheduler(s)
                alpha_t, var_t = gamma2as(gamma_t)

                z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

                noise_hat = model(z_t, mels, t_idx)
                loss, extra_dict = criterion(
                    base_noise_scheduler.gamma0,
                    base_noise_scheduler.gamma1,
                    (gamma_t - gamma_s) * train_T,
                    x, noise, noise_hat)
            else:
                t = torch.remainder(
                    uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
                t = t.clone().detach().requires_grad_(True)

                gamma_t = noise_scheduler(t)
                # gamma_t.retain_grad()

                alpha_t, var_t = gamma2as(gamma_t)
                z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

                noise_hat = model(z_t, mels, t.detach())
                d_gamma_t, *_ = grad(gamma_t.sum(), t,
                                     only_inputs=True, create_graph=True)
                loss, extra_dict = criterion(
                    base_noise_scheduler.gamma0,
                    base_noise_scheduler.gamma1,
                    d_gamma_t,
                    x, noise, noise_hat)

            # if minimize_var:
            #     loss_T_raw = extra_dict['loss_T_raw']
            #     loss.backward(retain_graph=True)
            #     gamma_t.backward(gradient=(2 * loss_T_raw - 1) * gamma_t.grad)
            # else:
        # loss.backward()
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        result = {'loss': loss.item()}
        result.update(extra_dict)
        return result

    trainer = Engine(process_function)

    ema_model = None
    if rank == 0:
        ema_handler = EMAHandler(model, momentum=0.0001)
        ema_model = ema_handler.ema_model
        ema_handler.attach(trainer, name="ema_momentum",
                           event=Events.ITERATION_COMPLETED)

        to_save = {
            'model': model,
            'ema_model': ema_model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'trainer': trainer,
            'noise_scheduler': noise_scheduler,
            'scaler': scaler
        }
    else:
        to_save = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'trainer': trainer,
            'noise_scheduler': noise_scheduler,
            'scaler': scaler
        }

    common.setup_common_training_handlers(
        trainer,
        train_sampler=train_sampler,
        to_save=to_save if rank == 0 else None,
        save_every_iters=eval_interval,
        output_path=save_dir,
        lr_scheduler=scheduler if not isinstance(
            scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None,
        output_names=['loss'] + extra_monitor,
        with_pbars=True if rank == 0 else False,
        with_pbar_on_iters=True,
        n_saved=2,
        log_every_iters=1,
        clear_cuda_cache=False
    )

    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, lambda engine: scheduler.step(
                engine.state.metrics['loss'])
        )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema_model' in to_save and 'ema_model' not in checkpoint:
            checkpoint['ema_model'] = checkpoint['model']
        Checkpoint.load_objects(
            to_load=to_save, checkpoint=checkpoint, strict=False)

    return trainer, ema_model


def training(local_rank, config: dict):
    rank = idist.get_rank()
    device = idist.device()

    print(rank, ": run with config:", config, "- backend=", idist.backend())
    print(f'world size = {idist.get_world_size()}')

    model_name = config['name']
    checkpoint_path = config['checkpoint']
    trainer_config = config['trainer']

    log_dir = trainer_config['log_dir']
    eval_file = trainer_config['eval_file']
    n_fft = trainer_config['n_fft']
    hop_length = trainer_config['hop_length']
    n_mels = trainer_config['n_mels']
    sr = trainer_config['sr']
    eval_interval = trainer_config['eval_interval']
    train_T = trainer_config['train_T']
    eval_T = trainer_config['eval_T']
    with_amp = trainer_config['with_amp']

    train_loader = get_dataflow(config)
    model, noise_scheduler, optimizer, scheduler = initialize(
        config, device)

    criterion = module_loss.diffusion_elbo

    mel_spec = module_arch.MelSpec(sr, n_fft, hop_length=hop_length,
                                   f_min=20, f_max=8000, n_mels=n_mels).to(device)
    # mel_spec = idist.auto_model(mel_spec)

    trainer, ema_model = create_trainer(model, mel_spec, noise_scheduler, optimizer,
                                        criterion, scheduler, device, trainer_config, train_loader.sampler,
                                        model_name, checkpoint_path)

    if rank == 0:
        # add model graph
        # use torchinfo
        for test_input in train_loader:
            break
        test_input = test_input[:1].to(device)
        test_mels = mel_spec(test_input)
        t = torch.tensor([0], device=device)
        summary(ema_model,
                input_data=(test_input, test_mels, t),
                device=device,
                col_names=("input_size", "output_size", "num_params", "kernel_size",
                           "mult_adds"),
                col_width=16,
                row_settings=("depth", "var_names"))

        tb_logger = get_logger(trainer, model, noise_scheduler, optimizer,
                               log_dir, model_name, eval_interval)

        eval_x, eval_sr = torchaudio.load(os.path.expanduser(eval_file))
        assert sr == eval_sr
        eval_x = eval_x.mean(0).to(device).unsqueeze(0)
        eval_mels = mel_spec(eval_x)

        @torch.no_grad()
        def predict_samples(engine):
            # model.eval()
            noise_scheduler.eval()

            # z_t = torch.randn(
            #     1, hop_length * (eval_mels.shape[-1] - 1) + 1, device=device)
            z_t = torch.randn_like(eval_x)

            with amp.autocast(enabled=with_amp):
                if train_T:
                    steps = torch.linspace(0, train_T, eval_T + 1,
                                           device=device).round().long()
                    gamma = noise_scheduler(steps / train_T)
                else:
                    steps = torch.linspace(0, 1, eval_T + 1, device=device)
                    gamma = noise_scheduler(steps)

                alpha, var = gamma2as(gamma)
                var_ts = - \
                    torch.expm1(F.softplus(gamma[:-1]) - F.softplus(gamma[1:]))
                print(alpha, var, var_ts)

                kld = 0
                for t in tqdm(range(eval_T, 0, -1)):
                    s = t - 1
                    noise_hat = ema_model(z_t, eval_mels, steps[t:t+1])
                    alpha_ts = alpha[t] / alpha[s]

                    noise = (z_t - alpha[t] * eval_x) * var[t].rsqrt()
                    kld += 0.5 * (gamma[t] - gamma[s]) * \
                        F.mse_loss(noise_hat, noise)

                    mu = (z_t - var_ts[s] * var[t].rsqrt()
                          * noise_hat) / alpha_ts
                    z_t = mu
                    if s:
                        z_t += (var_ts[s] * var[s] / var[t]).sqrt() * \
                            torch.rand_like(z_t)

                ll = -0.5 * (F.mse_loss(z_t, eval_x).log() +
                             1 + math.log(2 * math.pi)) - kld

            print("Log likelihood:", ll.item())

            predict = z_t.squeeze().clip(-0.99, 0.99)
            tb_logger.writer.add_audio(
                'predict', predict, engine.state.iteration, sample_rate=sr)

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
            every=eval_interval), predict_samples)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(
            every=eval_interval), plot_noise_curve)

    e = trainer.run(train_loader, max_epochs=1)

    if rank == 0:
        tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffWave')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='training checkpoint')

    args = parser.parse_args()

    config = json.load(open(args.config))
    validate(config, schema=CONFIG_SCHEMA)

    args_dict = vars(args)
    config.update(args_dict)

    backend = 'nccl'
    dist_configs = {
        'nproc_per_node': torch.cuda.device_count()
    }

    with idist.Parallel(backend=backend, **dist_configs) as parallel:
        parallel.run(training, config)
