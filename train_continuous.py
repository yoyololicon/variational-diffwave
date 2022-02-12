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
import os
from random import randrange, sample, uniform
from jsonschema import validate
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, EMAHandler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.engines import common
from ignite import distributed as idist
import torch_optimizer


from utils.schema import CONFIG_SCHEMA
from utils.utils import gamma2logas, get_instance, gamma2snr, snr2as, gamma2as
import models as module_arch
from inference import reverse_process_new

from train_distributed import get_dataflow


def initialize(config: dict, device):
    model = get_instance(module_arch, config['arch']).to(device)

    parameters = model.parameters()
    parameters = ContiguousParams(parameters)

    model = idist.auto_model(model)

    optim_args = config['optimizer']['args']
    try:
        optim_type = getattr(optim, config['optimizer']['type'])
    except AttributeError:
        optim_type = getattr(torch_optimizer, config['optimizer']['type'])
    optimizer = ZeroRedundancyOptimizer(
        parameters.contiguous(), optim_type, parameters_as_bucket_view=False, **optim_args)

    scheduler = get_instance(
        optim.lr_scheduler, config['lr_scheduler'], optimizer)

    return model, optimizer, scheduler


def create_trainer(model, mel_spec, noise_scheduler, optimizer: ZeroRedundancyOptimizer, scheduler, device, trainer_config, train_sampler, checkpoint_path: str):
    save_dir = trainer_config['save_dir']
    eval_interval = trainer_config['eval_interval']
    with_amp = trainer_config['with_amp']

    rank = idist.get_rank()

    scaler = amp.GradScaler(enabled=with_amp)

    def process_function(engine, batch):
        model.train()
        optimizer.zero_grad()

        x = batch
        x = x.to(device)
        noise = torch.randn_like(x)
        mels = mel_spec(x)

        N = x.shape[0]

        t = torch.remainder(
            uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
        gamma_t = t * noise_scheduler.gamma1 + \
            (1 - t) * noise_scheduler.gamma0

        with amp.autocast(enabled=with_amp):
            alpha, var = gamma2as(gamma_t)
            z_t = alpha[:, None] * x + var.sqrt()[:, None] * noise

            noise_hat = model(z_t, mels, t)
            loss = 0.5 * F.mse_loss(noise_hat, noise) * \
                (noise_scheduler.gamma1 - noise_scheduler.gamma0)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        result = {'loss': loss.item()}
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
            'scaler': scaler
        }
    else:
        to_save = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'trainer': trainer,
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
        output_names=['loss'],
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


def get_logger(trainer, model, optimizer, log_dir, model_name, interval):
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

    return tb_logger


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
    eval_T = trainer_config['eval_T']
    with_amp = trainer_config['with_amp']
    max_log_snr = trainer_config['max_log_snr']
    min_log_snr = trainer_config['min_log_snr']

    train_loader = get_dataflow(config)
    model, optimizer, scheduler = initialize(config, device)

    noise_scheduler = module_arch.CosineScheduler(
        gamma0=-max_log_snr, gamma1=-min_log_snr).to(device)

    mel_spec = module_arch.MelSpec(sr, n_fft, hop_length=hop_length,
                                   f_min=20, f_max=8000, n_mels=n_mels).to(device)

    trainer, ema_model = create_trainer(model, mel_spec, noise_scheduler, optimizer,
                                        scheduler, device, trainer_config, train_loader.sampler,
                                        checkpoint_path)

    if rank == 0:
        # add model graph
        # use torchinfo
        for test_input in train_loader:
            break
        test_input = test_input[:1].to(device)
        test_mels = mel_spec(test_input)
        t = torch.tensor([0.], device=device)
        summary(ema_model,
                input_data=(test_input, test_mels, t),
                device=device,
                col_names=("input_size", "output_size", "num_params", "kernel_size",
                           "mult_adds"),
                col_width=16,
                row_settings=("depth", "var_names"))

        tb_logger = get_logger(trainer, model, optimizer,
                               log_dir, model_name, eval_interval)

        eval_x, eval_sr = torchaudio.load(os.path.expanduser(eval_file))
        assert sr == eval_sr
        eval_x = eval_x.mean(0).to(device).unsqueeze(0)
        eval_mels = mel_spec(eval_x)

        @torch.no_grad()
        def predict_samples(engine):
            z_1 = torch.randn_like(eval_x)
            steps = torch.linspace(0, 1, eval_T + 1, device=device)
            gamma, steps = noise_scheduler(steps)

            z_0 = reverse_process_new(z_1, eval_mels, gamma,
                                      steps, ema_model, with_amp=with_amp)

            predict = z_0.squeeze().clip(-0.99, 0.99)
            tb_logger.writer.add_audio(
                'predict', predict, engine.state.iteration, sample_rate=sr)

        trainer.add_event_handler(Events.ITERATION_COMPLETED(
            every=eval_interval), predict_samples)

    e = trainer.run(train_loader, max_epochs=1)

    if rank == 0:
        tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='DiffWave Fixed-Noise Training')
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
