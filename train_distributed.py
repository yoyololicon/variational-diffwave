from torch.autograd import grad
import math
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.cuda import amp
from torch.distributed.optim import ZeroRedundancyOptimizer
from contiguous_params import ContiguousParams
import torchaudio
from torchinfo import summary
import argparse
import json
from datetime import datetime
from itertools import chain
import matplotlib.pyplot as plt
import os
from random import randrange, sample, uniform
from jsonschema import validate
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, EMAHandler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.engines import common
from ignite import distributed as idist
import torch_optimizer
from functools import partial


from utils.schema import CONFIG_SCHEMA
from utils.utils import gamma2logas, get_instance, gamma2snr, snr2as, gamma2as
import models as module_arch
import datasets as module_data
import loss as module_loss
from inference import reverse_process_new


def get_dataflow(config: dict):
    train_data = get_instance(module_data, config['dataset'])
    train_loader = idist.auto_dataloader(train_data, **config['data_loader'])
    return train_loader


def initialize(config: dict, device):
    model = get_instance(module_arch, config['arch']).to(device)
    noise_scheduler = module_arch.NoiseScheduler().to(device)

    parameters = chain(model.parameters(), noise_scheduler.parameters())
    parameters = ContiguousParams(parameters)

    model = idist.auto_model(model)
    noise_scheduler = idist.auto_model(noise_scheduler)

    optim_args = config['optimizer']['args']
    try:
        optim_type = getattr(optim, config['optimizer']['type'])
    except AttributeError:
        optim_type = getattr(torch_optimizer, config['optimizer']['type'])
    optimizer = ZeroRedundancyOptimizer(
        parameters.contiguous(), optim_type, parameters_as_bucket_view=False, **optim_args)
    # optimizer = idist.auto_optim(optimizer)

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


def create_trainer(model, mel_spec, noise_scheduler, optimizer: ZeroRedundancyOptimizer, criterion, scheduler, device, trainer_config, train_sampler, model_name: str, checkpoint_path: str):
    extra_monitor = trainer_config['extra_monitor']
    save_dir = trainer_config['save_dir']
    eval_interval = trainer_config['eval_interval']
    train_T = trainer_config['train_T']
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

        N = x.shape[0]
        if train_T:
            s = torch.remainder(
                uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
            s_idx = (s * train_T).long()
            t_idx = s_idx + 1

            t, s = t_idx / train_T, s_idx / train_T
            with amp.autocast(enabled=with_amp):
                gamma_ts, gamma_hat = noise_scheduler(torch.cat([t, s], dim=0))
                gamma_t, gamma_s = gamma_ts[:N], gamma_ts[N:]
                alpha_t, var_t = gamma2as(gamma_t)

                z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

                noise_hat = model(z_t, gamma_hat[:N], mels)

                loss, extra_dict = criterion(
                    base_noise_scheduler.gamma0,
                    base_noise_scheduler.gamma1,
                    torch.expm1(gamma_t - gamma_s) * train_T,
                    x, noise, noise_hat)
        else:
            t = torch.remainder(
                uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
            t = t.clone().detach().requires_grad_(True)

            with amp.autocast(enabled=with_amp):
                gamma_t, gamma_hat = noise_scheduler(t)
                gamma_hat.retain_grad()

                # alpha_t, var_t = gamma2as(gamma_t)
                log_alpha_t, log_var_t = gamma2logas(gamma_t)
                alpha_t, std_t = torch.exp(
                    log_alpha_t), torch.exp(log_var_t * 0.5)
                z_t = alpha_t[:, None] * x + std_t[:, None] * noise

                noise_hat = model(z_t, gamma_hat, mels)
                d_gamma_t, *_ = grad(gamma_t.sum(), t, create_graph=True)
                loss, extra_dict = criterion(
                    base_noise_scheduler.gamma0,
                    base_noise_scheduler.gamma1,
                    d_gamma_t,
                    x, noise, noise_hat)

                loss_T_raw = extra_dict['loss_T_raw']
                handle = gamma_hat.register_hook(
                    lambda grad: 2 * grad * loss_T_raw.to(grad.dtype))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if not train_T:
            handle.remove()
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

    @trainer.on(Events.ITERATION_COMPLETED(every=eval_interval))
    def consolidate_state_dict():
        optimizer.consolidate_state_dict()
        idist.barrier()

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

            z_1 = torch.randn_like(eval_x)

            if train_T:
                steps = torch.linspace(0, train_T, eval_T + 1,
                                       device=device).round().long()
                gamma, steps = noise_scheduler(steps / train_T)
            else:
                steps = torch.linspace(0, 1, eval_T + 1, device=device)
                gamma, steps = noise_scheduler(steps)

            infer_func = partial(ema_model, spectrogram=eval_mels)
            z_0 = reverse_process_new(z_1, gamma,
                                      steps, infer_func, with_amp=with_amp)

            predict = z_0.squeeze().clip(-0.99, 0.99)
            tb_logger.writer.add_audio(
                'predict', predict, engine.state.iteration, sample_rate=sr)

        @torch.no_grad()
        def plot_noise_curve(engine):
            figure = plt.figure()
            steps = torch.linspace(0, 1, 100, device=device)
            log_snr = -noise_scheduler(steps)[0].detach().cpu().numpy()
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
