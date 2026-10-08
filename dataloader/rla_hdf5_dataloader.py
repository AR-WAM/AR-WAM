"""Random image pairs from AR-WAM HDF5 episodes for RLA pretraining."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .hdf5_dataloader import _open_live_hdf5, _read_camera_frame


class RLAHDF5PairDataset(Dataset):
    def __init__(self, root, *, visual_cameras=('cam_head',), image_size=(320, 240),
                 stable_after_seconds=30., horizon=(2, 45), samples_per_epoch=1_000_000,
                 seed=2026):
        roots = (root,) if isinstance(root, (str, Path)) else tuple(root)
        cameras = tuple(visual_cameras)
        if len(cameras) != 1:
            raise ValueError('RLA pretraining requires one camera')
        self.camera = cameras[0]
        self.image_size = tuple(int(value) for value in image_size)
        self.horizon = tuple(int(value) for value in horizon)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError('image_size must contain positive width and height')
        if len(self.horizon) != 2 or not 2 <= self.horizon[0] <= self.horizon[1]:
            raise ValueError('horizon must contain inclusive bounds with minimum >= 2')
        self.samples_per_epoch = int(samples_per_epoch)
        if self.samples_per_epoch <= 0:
            raise ValueError('samples_per_epoch must be positive')
        self.seed = int(seed)
        self._rng = None
        self.episodes = []
        now = time.time()
        for root in roots:
            root = Path(root)
            for path in sorted(root.glob('*/aloha_agilex/data/episode_*.hdf5')):
                parts = [part.lower() for part in path.relative_to(root).parts]
                if any(part.startswith('.') or 'partial' in part or 'incomplete' in part for part in parts):
                    continue
                if stable_after_seconds > 0 and now - path.stat().st_mtime < stable_after_seconds:
                    continue
                with _open_live_hdf5(path) as handle:
                    stream = f'vision/{self.camera}/colors'
                    if stream not in handle:
                        raise ValueError(f'{path}: missing {stream}')
                    length = len(handle[stream])
                    if length < self.horizon[0]:
                        raise ValueError(f'{path}: RLA requires at least two frames and enough for the minimum horizon')
                self.episodes.append({'path': str(path), 'length': length})
        if not self.episodes:
            raise ValueError('no stable HDF5 episodes found in dataset_dirs')

    def __len__(self):
        return self.samples_per_epoch

    def _sampling_rng(self):
        if self._rng is None:
            worker = get_worker_info()
            rank = int(os.environ.get('RANK', '0'))
            worker_seed = int(worker.seed) if worker is not None else 0
            self._rng = np.random.default_rng((self.seed + worker_seed + rank * 1_000_003) % (2**63 - 1))
        return self._rng

    def __getitem__(self, index):
        rng = self._sampling_rng()
        episode = self.episodes[int(rng.integers(len(self.episodes)))]
        length = episode['length']
        start = int(rng.integers(length - self.horizon[0] + 1))
        horizon = int(rng.integers(self.horizon[0], min(self.horizon[1], length - start) + 1))
        end = start + horizon - 1
        frames = []
        with _open_live_hdf5(Path(episode['path'])) as handle:
            for frame_index in (start, end):
                frame = _read_camera_frame(handle, self.camera, frame_index)
                expected = (self.image_size[1], self.image_size[0], 3)
                if frame.shape != expected:
                    raise ValueError(f"{episode['path']}: image shape {frame.shape} does not match {expected}")
                frames.append(frame.transpose(2, 0, 1).astype(np.float32) / 255.)
        return {
            'images': torch.from_numpy(np.stack(frames)[:, None]),
            'frame_indices': torch.tensor([start, end], dtype=torch.long),
            'horizon': torch.tensor(horizon, dtype=torch.long),
        }
