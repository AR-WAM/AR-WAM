"""Train the RoboTwin RLA encoder/decoder consumed by AR-WAM."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.rla_hdf5_dataloader import RLAHDF5PairDataset
from models.model_runner import ModelFactory
from models.rla_wm.rla_autoencoder import FrozenDINOFeatures, RlaAutoencoder, reconstruction_loss
from train import _per_rank_batch_size, _setup_distributed


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(output_dir, model, optimizer, step, config, scaler):
    """Export plain FP32 encoder/decoder weights plus resumable training state."""
    module = model.module if isinstance(model, DistributedDataParallel) else model
    directory = Path(output_dir) / 'ckpts'
    directory.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    for role in ('encoder', 'decoder'):
        prefix = role + '.'
        weights = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
        _atomic_save(weights, directory / f'{role}_step{step:07d}.pt')
    checkpoint = {'format': 'rla_training_v2', 'step': step, 'config': config,
                  'model': state, 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                  'world_size': int(os.environ.get('WORLD_SIZE', '1'))}
    _atomic_save(checkpoint, directory / f'training_step{step:07d}.pt')
    _atomic_save(checkpoint, directory / 'latest.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    training = config['training']
    batch_size = _per_rank_batch_size(training, int(os.environ.get('WORLD_SIZE', '1')))
    precision = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    amp_dtype = precision[training['mixed_precision']]
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['format'] != 'rla_training_v2':
            raise ValueError('resume requires a train_rla.py training checkpoint')
        if checkpoint.get('world_size') != int(os.environ.get('WORLD_SIZE', '1')):
            raise ValueError('RLA resume requires the same world_size and topology metadata')
        runtime_keys = {'max_steps', 'log_interval', 'save_interval'}
        saved = dict(checkpoint['config'])
        current = dict(config)
        for recipe in (saved, current):
            recipe.pop('system', None)
            recipe['training'] = {key: value for key, value in recipe['training'].items()
                                  if key not in runtime_keys}
        if saved != current:
            raise ValueError('RLA resume settings differ from the checkpoint')
    rank, local_rank, world_size, device = _setup_distributed()
    cv2.setNumThreads(1)
    torch.manual_seed(rank)
    np.random.seed(rank)
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        print(f'global_batch_size={training["global_batch_size"]} world_size={world_size} per_gpu_batch_size={batch_size}', flush=True)
    dataset_config = config['dataset']
    dataset = RLAHDF5PairDataset(
        dataset_config['dataset_dirs'], visual_cameras=dataset_config['camera_names'],
        image_size=dataset_config['image_size'],
        stable_after_seconds=float(dataset_config['stable_after_seconds']),
        horizon=dataset_config['horizon'], samples_per_epoch=dataset_config['samples_per_epoch'],
        seed=dataset_config['seed'],
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    workers = int(config['system']['num_workers'])
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0, drop_last=True)
    if len(loader) == 0:
        raise ValueError('RLA dataset must provide at least one full batch per rank')
    model = RlaAutoencoder(config['models']).to(device=device, dtype=torch.float32)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = torch.optim.AdamW(module.parameters(), lr=float(training['learning_rate']),
                                  weight_decay=float(training['weight_decay']),
                                  betas=tuple(training['betas']), eps=float(training['eps']))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    step = 0
    if checkpoint is not None:
        module.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        step = int(checkpoint['step'])
        del checkpoint
    if rank == 0:
        config_path = output / 'config.yaml'
        temporary = config_path.with_suffix('.yaml.tmp')
        try:
            temporary.write_text(yaml.safe_dump(config, sort_keys=False))
            os.replace(temporary, config_path)
        finally:
            temporary.unlink(missing_ok=True)
    vision, _, _, _ = ModelFactory.create_vision_encoder(config['vision_encoder']['checkpoint_path'],
                                                        dtype=torch.float32, device=device)
    features = FrozenDINOFeatures(vision).to(device)
    max_steps = int(args.max_steps if args.max_steps is not None else training['max_steps'])
    epoch = 0
    model.train()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            images = batch['images'].to(device, non_blocking=True)
            current, future = features(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                prediction = model(current, future)
                loss, info = reconstruction_loss(prediction.float(), future.float(),
                                                 training['lambda_l1'], training['lambda_mse'])
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = scaler.get_scale() >= previous_scale
            step += 1
            if rank == 0 and (step == 1 or step % int(training['log_interval']) == 0):
                print(f'step={step} loss={float(loss):.6f} loss_l1={float(info["loss_l1"]):.6f} '
                      f'loss_mse={float(info["loss_mse"]):.6f} scale={scaler.get_scale():.0f} updated={updated}', flush=True)
            if rank == 0 and (step % int(training['save_interval']) == 0 or step == max_steps):
                save_checkpoint(output, model, optimizer, step, config, scaler)
            if step >= max_steps:
                break
        epoch += 1
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
