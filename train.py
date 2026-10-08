"""Structured AdaLN training on RoboTwin HDF5 episodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataloader.hdf5_dataloader import RLAHDF5Dataset
from models.optimizer import FP32AdamW
from models.model_runner import ModelFactory, VLAWrapper


CHECKPOINT_FORMAT = "ar_wam_style_v1"


def build_lr_scheduler(optimizer, training):
    config = training["lr_scheduler"]
    if config["type"] != "cosine":
        raise ValueError("this training recipe supports only a cosine LR scheduler")
    return CosineAnnealingLR(
        optimizer.optimizer,
        T_max=int(config["total_steps"]),
        eta_min=float(config["min_lr"]),
    )


def build_train_config(config):
    training = config["training"]
    keys = (
        "time_mu", "time_sigma",
        "lambda_action", "lambda_rla", "lambda_mask", "lambda_mask_token",
        "lambda_condition_decode", "lambda_run_endpoint_state",
    )
    return {key: training[key] for key in keys}


def _per_rank_batch_size(training, world_size):
    global_batch = training["global_batch_size"]
    if (not isinstance(global_batch, int) or isinstance(global_batch, bool)
            or global_batch <= 0 or world_size <= 0 or global_batch % world_size):
        raise ValueError("global_batch_size must be a positive integer divisible by WORLD_SIZE")
    return global_batch // world_size


def _setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def _load_model_warm_start(model, checkpoint):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    module.load_state_dict(checkpoint["model"], strict=True)


def _resume_config(config):
    """Settings that must agree before restoring optimizer/scheduler state."""
    runtime_keys = {"max_steps", "output_dir", "log_interval", "save_interval", "latest_interval"}
    return {
        "common": config.get("common"),
        "model": config.get("model"),
        "dataset": config.get("dataset"),
        "training": {key: value for key, value in config["training"].items() if key not in runtime_keys},
    }


def _validate_resume(checkpoint, config, norm_stats, world_size):
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("resume requires a checkpoint with the current style schema and normalization metadata; "
                         "use a checkpoint from this implementation")
    saved = _resume_config(checkpoint["config"])
    current = _resume_config(config)
    changed = [key for key in current if current[key] != saved[key]]
    if changed:
        raise ValueError(f"resume settings differ in {changed}; use --warm-start for a new recipe")
    if checkpoint.get("norm_stats") != norm_stats:
        raise ValueError("resume normalization statistics differ from the checkpoint")
    if checkpoint.get("world_size") != world_size:
        raise ValueError("resume world_size differs from the checkpoint (distributed topology changes)")


def _save_checkpoint(path, model, optimizer, step, config, scheduler, norm_stats=None, world_size=1):
    module = model.module if isinstance(model, DistributedDataParallel) else model
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save({
            "format": CHECKPOINT_FORMAT,
            "step": step,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "norm_stats": norm_stats,
            "world_size": world_size,
        }, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume")
    initialization.add_argument("--warm-start")
    args = parser.parse_args()

    cv2.setNumThreads(1)
    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    training = config["training"]
    batch_size = _per_rank_batch_size(training, int(os.environ.get("WORLD_SIZE", "1")))
    with open(args.norm_stats) as handle:
        norm_stats = json.load(handle)["robotwin2"]
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        _validate_resume(checkpoint, config, norm_stats, int(os.environ.get("WORLD_SIZE", "1")))
    rank, local_rank, world_size, device = _setup_distributed()
    output_dir = Path(args.output_dir or training["output_dir"])
    if rank == 0:
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"global_batch_size={training['global_batch_size']} "
              f"world_size={world_size} per_gpu_batch_size={batch_size}", flush=True)

    dataset_config = config["dataset"]
    dataset = RLAHDF5Dataset(
        dataset_config["dataset_dirs"],
        visual_cameras=dataset_config["camera_names"],
        image_size=dataset_config["image_size"],
        stable_after_seconds=float(dataset_config["stable_after_seconds"]),
        bbox_jitter=dataset_config["bbox_jitter"],
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    workers = int(config["system"]["num_workers"])
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=sampler is None, sampler=sampler, num_workers=workers,
        pin_memory=True, persistent_workers=workers > 0, drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("dataset must provide at least one full batch per rank")

    vision_config = config["model"]["vision_encoder"]
    vision, dino_dim, registers, patch = ModelFactory.create_vision_encoder(
        vision_config["checkpoint_path"], dtype=torch.float32, device=device,
    )
    model = ModelFactory.create_action_model(
        _ConfigProxy(config), dino_dim, len(vision_config["feat_layers"]), patch_size=patch,
    ).to(device=device, dtype=torch.float32)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                        find_unused_parameters=True)
    future = config["model"]["future_feat"]
    wrapper = VLAWrapper(
        vision, model, training["time_sampler"], vision_config["feat_layers"],
        True, registers, device, torch.float32, args.norm_stats,
        train_config=build_train_config(config),
        rla_work_dir=future["rla_work_dir"], rla_checkpoint_step=future["checkpoint_step"],
    ).to(device)
    module = model.module if isinstance(model, DistributedDataParallel) else model
    optimizer = FP32AdamW(
        module.named_parameters(), lr=float(training["learning_rate"]),
        betas=tuple(training["betas"]), weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_lr_scheduler(optimizer, training)
    step = 0
    if args.warm_start:
        _load_model_warm_start(model, torch.load(args.warm_start, map_location="cpu", weights_only=True))
    if checkpoint is not None:
        module.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = int(checkpoint["step"])
        if scheduler.last_epoch != step:
            raise ValueError("checkpoint step and scheduler epoch differ")
        del checkpoint

    max_steps = int(args.max_steps if args.max_steps is not None else training["max_steps"])
    epoch = 0
    model.train()
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, info = wrapper(batch)
            optimizer.backward(loss.float())
            _, grad_norm = optimizer.prepare_grads(max_norm=float(training["grad_clip_norm"]))
            if optimizer.step():
                step += 1
                scheduler.step()
            if rank == 0 and (step == 1 or step % int(training["log_interval"]) == 0):
                losses = " ".join(f"{key}={float(info[key]):.6f}" for key in (
                    "loss_action", "loss_rla", "loss_mask_token", "loss_mask",
                    "loss_condition_decode", "loss_run_endpoint_state",
                ))
                print(f"step={step} loss={float(loss):.6f} lr={scheduler.get_last_lr()[0]:.10f} "
                      f"grad_norm={float(grad_norm):.6f} {losses}", flush=True)
            if rank == 0 and (step % int(training["save_interval"]) == 0 or step == max_steps):
                _save_checkpoint(output_dir / "checkpoints" / f"step_{step:07d}.pt",
                                 model, optimizer, step, config, scheduler, norm_stats, world_size)
            if rank == 0 and (step % int(training["latest_interval"]) == 0 or step == max_steps):
                _save_checkpoint(output_dir / "checkpoints/latest.pt",
                                 model, optimizer, step, config, scheduler, norm_stats, world_size)
            if step >= max_steps:
                break
        epoch += 1
    if world_size > 1:
        dist.destroy_process_group()


class _ConfigProxy:
    """Attribute access facade for ModelFactory's existing config contract."""
    def __init__(self, value):
        self._value = value

    def __getattr__(self, key):
        value = self._value[key]
        if isinstance(value, dict):
            return _ConfigProxy(value)
        return value

    def get(self, key, default=None):
        value = self._value.get(key, default)
        return _ConfigProxy(value) if isinstance(value, dict) else value


if __name__ == "__main__":
    main()
