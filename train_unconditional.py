import torch
from torch import optim, nn
from torch.autograd import grad
from torch.cuda import amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchaudio
from torchinfo import summary
import argparse
import json
import matplotlib.pyplot as plt
import os
from random import randrange, sample, uniform
from jsonschema import validate
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, EMAHandler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.engines import common
from ignite import distributed as idist


from utils.schema import CONFIG_SCHEMA
from utils.utils import gamma2logas, get_instance, gamma2snr, snr2as, gamma2as
import loss as module_loss
from inference import reverse_process_new

from train_distributed import get_dataflow, initialize, get_logger


def create_trainer(model, noise_scheduler, optimizer: ZeroRedundancyOptimizer, criterion, scheduler, device, trainer_config, train_sampler, checkpoint_path: str):
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

        x, *c = batch
        x = x.to(device)
        c = tuple(c_i.to(device) for c_i in c)
        noise = torch.randn_like(x)

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

                noise_hat = model(z_t, gamma_hat[:N], *c)

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

                log_alpha_t, log_var_t = gamma2logas(gamma_t)
                alpha_t, std_t = torch.exp(
                    log_alpha_t), torch.exp(log_var_t * 0.5)
                z_t = alpha_t[:, None] * x + std_t[:, None] * noise

                noise_hat = model(z_t, gamma_hat, *c)

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
    eval_dur = trainer_config['eval_dur']
    sr = trainer_config['sr']
    eval_interval = trainer_config['eval_interval']
    eval_T = trainer_config['eval_T']
    with_amp = trainer_config['with_amp']

    train_loader = get_dataflow(config)
    model, noise_scheduler, optimizer, scheduler = initialize(
        config, device)

    criterion = module_loss.diffusion_elbo

    trainer, ema_model = create_trainer(model, noise_scheduler, optimizer, criterion,
                                        scheduler, device, trainer_config, train_loader.sampler,
                                        checkpoint_path)

    if rank == 0:
        # add model graph
        # use torchinfo
        for test_input in train_loader:
            break
        # test_input = test_input[:1].to(device)
        x, c = test_input
        x = x[:1].to(device)
        c = c[:1].to(device)
        t = torch.tensor([0.], device=device)
        summary(ema_model,
                input_data=(x, t, c),
                device=device,
                col_names=("input_size", "output_size", "num_params", "kernel_size",
                           "mult_adds"),
                col_width=16,
                row_settings=("depth", "var_names"))

        tb_logger = get_logger(trainer, model, noise_scheduler, optimizer,
                               log_dir, model_name, eval_interval)

        @torch.no_grad()
        def generate_samples(engine):
            z_1 = torch.randn(1, sr * eval_dur, device=device)
            steps = torch.linspace(0, 1, eval_T + 1, device=device)
            gamma, steps = noise_scheduler(steps)

            z_0 = reverse_process_new(z_1, gamma,
                                      steps, ema_model, with_amp=with_amp)

            generated = z_0.squeeze().clip(-0.99, 0.99)
            tb_logger.writer.add_audio(
                'generated', generated, engine.state.iteration, sample_rate=sr)

        @torch.no_grad()
        def plot_noise_curve(engine):
            figure = plt.figure()
            steps = torch.linspace(0, 1, 100, device=device)
            log_snr = -noise_scheduler(steps)[0].detach().cpu().numpy()
            steps = steps.cpu().numpy()
            plt.plot(steps, log_snr)
            tb_logger.writer.add_figure(
                'log_snr', figure, engine.state.iteration)

        # trainer.add_event_handler(Events.ITERATION_COMPLETED(
        #     every=eval_interval), generate_samples)
        trainer.add_event_handler(Events.ITERATION_COMPLETED(
            every=eval_interval), plot_noise_curve)

    e = trainer.run(train_loader, max_epochs=1)

    if rank == 0:
        tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unconditional DiffWave Fixed-Noise Training')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='training checkpoint')

    args = parser.parse_args()

    config = json.load(open(args.config))
    # validate(config, schema=CONFIG_SCHEMA)

    args_dict = vars(args)
    config.update(args_dict)

    backend = 'nccl'
    dist_configs = {
        'nproc_per_node': torch.cuda.device_count()
    }

    with idist.Parallel(backend=backend, **dist_configs) as parallel:
        parallel.run(training, config)
