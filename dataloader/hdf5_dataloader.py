"""RoboTwin HDF5 frame candidates with terminal-held action/future targets."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

RAW_SKILL_NAMES = {
    0: "idle", 1: "grasp", 2: "place", 3: "click", 4: "bimanual_same",
    5: "bimanual_two", 6: "handover", 7: "shake", 8: "pour", 9: "hang",
    10: "pull", 11: "open_lid", 12: "switch_on", 13: "beat", 14: "scan",
    15: "rotate", 16: "homing", 17: "postplace", 18: "lift", 19: "retract",
    20: "pre_handover", 21: "prepare_skillet", 22: "remove",
}
RAW_FIELDS = (
    "skill", "participants", "style", "kind_l", "kind_r", "valid_l",
    "valid_r", "bbox_l", "bbox_r",
)
RUN_KEY_FIELDS = ("skill", "participants")

def _decode_jpeg(value: Any) -> np.ndarray:
    raw = value.tobytes() if isinstance(value, np.ndarray) else bytes(value)
    image = cv2.imdecode(np.frombuffer(raw.rstrip(b"\0"), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode HDF5 JPEG")
    return image


def _open_live_hdf5(path: Path) -> h5py.File:
    """Open a live episode, tolerating a brief atomic file replacement."""
    for attempt in range(20):
        try:
            return h5py.File(path, "r")
        except OSError as error:
            message = str(error).lower()
            transient = isinstance(error, FileNotFoundError) or any(
                marker in message
                for marker in ("file signature not found", "bad object header", "bad superblock")
            )
            if not transient or attempt == 19:
                raise
            time.sleep(1.0)


def _read_camera_frame(h: h5py.File, camera: str, index: int) -> np.ndarray:
    path = f"vision/{camera}/colors"
    if path not in h:
        raise KeyError(f"requested camera stream is missing: {path}")
    dataset = h[path]
    value = dataset[int(index)]
    if isinstance(value, np.ndarray) and value.ndim == 3:
        return value.astype(np.uint8, copy=False)
    return _decode_jpeg(value)


def _decode_v4_mask(mask: np.ndarray) -> np.ndarray:
    """Decode v4 colors to right/left grasp, carry, and place masks."""
    if mask.ndim != 3 or mask.shape[-1] != 3:
        raise ValueError(f"v4 mask must have shape (H,W,3), got {mask.shape}")
    component_values = (
        (127, 197, 227, 41),  # grasp
        (100, 170, 227, 41),  # carry
        (70, 170, 197, 41),   # place
    )
    right = [np.isin(mask[..., 0], sums) for sums in component_values]
    both = [np.isin(mask[..., 1], sums) for sums in component_values]
    left = [np.isin(mask[..., 2], sums) for sums in component_values]
    channels = [
        *(hand | shared for hand, shared in zip(right, both)),
        *(hand | shared for hand, shared in zip(left, both)),
    ]
    return np.stack(channels, axis=0).astype(np.uint8)


def _read_camera_mask_raw(h: h5py.File, camera: str, index: int) -> np.ndarray:
    path = f"vision/{camera}/mask"
    if path not in h:
        raise KeyError(f"required mask stream is missing: {path}")
    return np.asarray(h[path][int(index)], dtype=np.uint8)


def _read_camera_mask(h: h5py.File, camera: str, index: int) -> np.ndarray:
    raw = _read_camera_mask_raw(h, camera, index)
    return _decode_v4_mask(raw)


def _box(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(4)
    except (TypeError, ValueError):
        return np.zeros(4, dtype=np.float32)
    if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1) or arr[2] <= 0 or arr[3] <= 0:
        return np.zeros(4, dtype=np.float32)
    return arr


def bbox_iou_cxcyhw(a: np.ndarray, b: np.ndarray) -> float:
    """Return IoU for two normalized ``(cx, cy, h, w)`` boxes."""
    a = np.asarray(a, dtype=np.float32).reshape(4)
    b = np.asarray(b, dtype=np.float32).reshape(4)
    if min(float(a[2]), float(a[3]), float(b[2]), float(b[3])) <= 0:
        return 0.0
    ax1, ay1 = a[0] - a[3] / 2, a[1] - a[2] / 2
    ax2, ay2 = a[0] + a[3] / 2, a[1] + a[2] / 2
    bx1, by1 = b[0] - b[3] / 2, b[1] - b[2] / 2
    bx2, by2 = b[0] + b[3] / 2, b[1] + b[2] / 2
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = float(a[2] * a[3] + b[2] * b[3] - intersection)
    return float(intersection / union) if union > 0 else 0.0


def jitter_bbox_cxcyhw(
    box: np.ndarray,
    *,
    rng=None,
    scale_range=(0.9, 1.1),
    center_shift_fraction=0.05,
    min_iou=0.8,
    max_attempts=1,
) -> np.ndarray:
    """Apply bounded detector-like jitter while preserving the image and IoU."""
    original = np.asarray(box, dtype=np.float32).reshape(4)
    if original[2] <= 0 or original[3] <= 0:
        return original.copy()
    rng = np.random if rng is None else rng
    scale_min, scale_max = (float(value) for value in scale_range)
    for _ in range(max(1, int(max_attempts))):
        height = min(
            1.0,
            float(original[2]) * float(rng.uniform(scale_min, scale_max)),
        )
        width = min(
            1.0,
            float(original[3]) * float(rng.uniform(scale_min, scale_max)),
        )
        cx = float(original[0]) + float(
            rng.uniform(-center_shift_fraction, center_shift_fraction)
        ) * float(original[3])
        cy = float(original[1]) + float(
            rng.uniform(-center_shift_fraction, center_shift_fraction)
        ) * float(original[2])
        if not (
            width / 2 <= cx <= 1.0 - width / 2
            and height / 2 <= cy <= 1.0 - height / 2
        ):
            continue
        augmented = np.asarray([cx, cy, height, width], dtype=np.float32)
        if bbox_iou_cxcyhw(original, augmented) >= float(min_iou):
            return augmented
    return original.copy()


def jitter_current_bboxes(
    bbox_l: np.ndarray,
    bbox_r: np.ndarray,
    *,
    valid_l: bool,
    valid_r: bool,
    probability: float,
    rng=None,
    scale_range=(0.9, 1.1),
    center_shift_fraction=0.05,
    min_iou=0.8,
    max_attempts=1,
) -> tuple[np.ndarray, np.ndarray]:
    """Jitter current valid slots, sharing noise for equal two-hand boxes."""
    left = np.asarray(bbox_l, dtype=np.float32).reshape(4).copy()
    right = np.asarray(bbox_r, dtype=np.float32).reshape(4).copy()
    rng = np.random if rng is None else rng
    if float(probability) <= 0 or float(rng.random()) >= float(probability):
        return left, right
    kwargs = {
        "rng": rng,
        "scale_range": scale_range,
        "center_shift_fraction": center_shift_fraction,
        "min_iou": min_iou,
        "max_attempts": max_attempts,
    }
    if valid_l and valid_r and np.allclose(left, right, atol=1e-7, rtol=0):
        shared = jitter_bbox_cxcyhw(left, **kwargs)
        return shared, shared.copy()
    if valid_l:
        left = jitter_bbox_cxcyhw(left, **kwargs)
    if valid_r:
        right = jitter_bbox_cxcyhw(right, **kwargs)
    return left, right


def _signature(arrays: Mapping[str, np.ndarray], index: int) -> tuple[Any, ...]:
    result = []
    for field in RUN_KEY_FIELDS:
        result.extend(np.asarray(arrays[field][index]).reshape(-1).tolist())
    return tuple(result)


def extract_raw_runs(arrays: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split runs on skill/participant changes only.

    Style and slot kind/validity/bbox are frame-level details
    that may change while the same skill is executed by the same arm.  They are
    intentionally not part of the run boundary signature.
    """
    missing = [field for field in RAW_FIELDS if field not in arrays]
    if missing:
        raise ValueError(f"missing raw atomic fields: {missing}")
    source_arrays = {field: np.asarray(value) for field, value in arrays.items()}
    arrays = {field: source_arrays[field] for field in RAW_FIELDS}
    lengths = {field: int(value.shape[0]) for field, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"raw annotation arrays must have the same length: {lengths}")
    total = next(iter(lengths.values()), 0)
    if not total:
        return []
    boundaries = [0]
    previous = _signature(arrays, 0)
    for index in range(1, total):
        current = _signature(arrays, index)
        if current != previous:
            boundaries.append(index)
            previous = current
    boundaries.append(total)
    runs = []
    for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
        i = t0
        skill_id = int(np.asarray(arrays["skill"][i]).item())
        bbox_l = _box(arrays["bbox_l"][i])
        bbox_r = _box(arrays["bbox_r"][i])
        runs.append({
            "t0": t0, "t1": t1, "skill_id": skill_id,
            "skill_name": RAW_SKILL_NAMES.get(skill_id, f"unknown_{skill_id}"),
            "participants_id": int(np.asarray(arrays["participants"][i]).item()),
            "style_id": int(np.asarray(arrays["style"][i]).item()),
            "kind_l": int(np.asarray(arrays["kind_l"][i]).item()), "kind_r": int(np.asarray(arrays["kind_r"][i]).item()),
            "valid_l": int(np.asarray(arrays["valid_l"][i]).item()), "valid_r": int(np.asarray(arrays["valid_r"][i]).item()),
            "bbox_l": bbox_l, "bbox_r": bbox_r,
            "bbox_l_is_none": bool(np.all(bbox_l == 0)),
            "bbox_r_is_none": bool(np.all(bbox_r == 0)),
        })
    return runs


def _query(anchor: int, offsets: np.ndarray, t0: int, t1: int, total: int):
    raw = anchor + offsets
    last = max(t0, min(t1 - 1, total - 1))
    valid = (raw >= t0) & (raw < t1) & (raw >= 0) & (raw < total)
    return np.clip(raw, t0, last), valid


def _read_atomic_frame(h: h5py.File, index: int) -> dict[str, Any]:
    """Read model-facing atomic labels at one episode-local frame."""
    atomic = h["annotation/atomic"]
    i = int(index)
    bbox_l = _box(atomic["bbox_l"][i])
    bbox_r = _box(atomic["bbox_r"][i])
    return {
        "skill_id": int(np.asarray(atomic["skill"][i]).item()),
        "participants_id": int(np.asarray(atomic["participants"][i]).item()),
        "style_id": int(np.asarray(atomic["style"][i]).item()),
        "kind_l": int(np.asarray(atomic["kind_l"][i]).item()),
        "kind_r": int(np.asarray(atomic["kind_r"][i]).item()),
        "valid_l": int(np.asarray(atomic["valid_l"][i]).item()),
        "valid_r": int(np.asarray(atomic["valid_r"][i]).item()),
        "bbox_l": bbox_l,
        "bbox_r": bbox_r,
        "bbox_l_is_none": bool(np.all(bbox_l == 0)),
        "bbox_r_is_none": bool(np.all(bbox_r == 0)),
    }


def _validate_episode(h: h5py.File, camera: str, image_size: tuple[int, int]) -> dict:
    """Reject malformed training streams before building frame candidates."""
    length = None
    for name, width in (("state/ee_states", 16), ("action/joint_states", 14)):
        if name not in h:
            raise ValueError(f"{h.filename}: missing {name}")
        values = h[name][:]
        if values.ndim != 2 or values.shape[1] != width or not len(values):
            raise ValueError(f"{h.filename}: {name} must have shape (T,{width}), T > 0; got {values.shape}")
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise ValueError(f"{h.filename}: {name} must contain finite numeric values")
        if length is not None and len(values) != length:
            raise ValueError(f"{h.filename}: {name} length must equal state length {length}")
        length = len(values)
    if "annotation/atomic" not in h:
        raise ValueError(f"{h.filename}: missing annotation/atomic")
    group = h["annotation/atomic"]
    layout = group.attrs.get("bbox_layout", "")
    if isinstance(layout, bytes):
        layout = layout.decode("utf-8")
    if layout != "cxcyhw_01":
        raise ValueError(f"{h.filename}: unsupported bbox_layout: {layout!r}")
    atomic = {}
    limits = {"skill": 23, "participants": 4, "style": 5,
              "kind_l": 3, "kind_r": 3, "valid_l": 2, "valid_r": 2}
    for field in RAW_FIELDS:
        if field not in group:
            raise ValueError(f"{h.filename}: missing annotation/atomic/{field}")
        values = group[field][:]
        if field in limits:
            if values.shape not in ((length,), (length, 1)):
                raise ValueError(f"{h.filename}: {field} must have shape (T,) or (T,1), T={length}; got {values.shape}")
            values = values.reshape(length)
            if (not np.issubdtype(values.dtype, np.number)
                    or not np.isfinite(values).all() or np.any(values != np.floor(values))
                    or np.any(values < 0) or np.any(values >= limits[field])):
                raise ValueError(f"{h.filename}: {field} must contain integer IDs in [0,{limits[field]-1}]")
        elif values.shape != (length, 4):
            raise ValueError(f"{h.filename}: {field} must have shape ({length},4); got {values.shape}")
        atomic[field] = values
    for arm in ("l", "r"):
        boxes = atomic[f"bbox_{arm}"]
        if not np.issubdtype(boxes.dtype, np.number):
            raise ValueError(f"{h.filename}: bbox_{arm} must contain numeric values")
        active = atomic[f"valid_{arm}"] == 1
        invalid = (~np.isfinite(boxes).all(axis=1) | (boxes < 0).any(axis=1)
                   | (boxes > 1).any(axis=1) | (boxes[:, 2:] <= 0).any(axis=1))
        if np.any(active & invalid):
            raise ValueError(f"{h.filename}: valid bbox_{arm} must be finite normalized cxcyhw with positive size")
    for stream in ("colors", "mask"):
        name = f"vision/{camera}/{stream}"
        if name not in h:
            raise ValueError(f"{h.filename}: missing required {name}")
        dataset = h[name]
        if not dataset.shape or dataset.shape[0] != length:
            raise ValueError(f"{h.filename}: {name} length must equal state length {length}")
        expected = (length, image_size[1], image_size[0], 3)
        if stream == "mask" or dataset.ndim != 1:
            if dataset.shape != expected or dataset.dtype != np.uint8:
                raise ValueError(f"{h.filename}: {name} must be uint8 {expected}; got {dataset.shape}, {dataset.dtype}")
    return atomic


class RLAHDF5Dataset(Dataset):
    """HDF5 dataset exposing each frame as a fixed in-run anchor."""

    def __init__(self, root: str | Path | Sequence[str | Path], *, action_horizon: int = 32,
                 state_history: int = 1, visual_cameras: Sequence[str] = ("cam_head",),
                 stable_after_seconds: float = 30.0,
                 image_size: Sequence[int] = (320, 240),
                 bbox_jitter: Mapping[str, Any] | None = None):
        if isinstance(root, (str, Path)):
            self.roots = (Path(root),)
        else:
            self.roots = tuple(Path(value) for value in root)
        if not self.roots:
            raise ValueError("RLAHDF5Dataset requires at least one dataset root")
        self.root = self.roots[0] if len(self.roots) == 1 else self.roots
        self.action_horizon = int(action_horizon)
        if self.action_horizon != 32:
            raise ValueError("RLAHDF5Dataset requires a 32-step action horizon")
        self.state_history = int(state_history)
        self.future_horizon = 1
        self.visual_cameras = tuple(visual_cameras)
        if self.state_history != 1 or len(self.visual_cameras) != 1:
            raise ValueError("training requires one state frame and one camera")
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive width and height")
        jitter = dict(bbox_jitter or {})
        self.bbox_jitter_enabled = bool(jitter.get("enabled", False))
        self.bbox_jitter_probability = float(jitter.get("probability", 0.0))
        self.bbox_jitter_scale_range = tuple(
            float(value) for value in jitter.get("scale_range", (0.9, 1.1))
        )
        self.bbox_jitter_center_shift_fraction = float(
            jitter.get("center_shift_fraction", 0.05)
        )
        self.bbox_jitter_min_iou = float(jitter.get("min_iou", 0.8))
        self.bbox_jitter_max_attempts = int(jitter.get("max_attempts", 1))
        if not 0.0 <= self.bbox_jitter_probability <= 1.0:
            raise ValueError("bbox jitter probability must be in [0, 1]")
        if len(self.bbox_jitter_scale_range) != 2 or not (
            0 < self.bbox_jitter_scale_range[0]
            <= self.bbox_jitter_scale_range[1]
        ):
            raise ValueError("bbox jitter scale_range must contain two positive bounds")
        if self.bbox_jitter_center_shift_fraction < 0:
            raise ValueError("bbox jitter center_shift_fraction must be nonnegative")
        if not 0.0 <= self.bbox_jitter_min_iou <= 1.0:
            raise ValueError("bbox jitter min_iou must be in [0, 1]")
        if self.bbox_jitter_max_attempts <= 0:
            raise ValueError("bbox jitter max_attempts must be positive")
        self.samples: list[dict[str, Any]] = []
        self.episodes: list[dict[str, Any]] = []
        self.skipped_paths: list[str] = []
        now = time.time()
        for path in self._paths():
            if stable_after_seconds > 0 and now - path.stat().st_mtime < stable_after_seconds:
                continue
            try:
                with _open_live_hdf5(path) as h:
                    atomic = _validate_episode(h, self.visual_cameras[0], self.image_size)
                    state_length = int(h["state/ee_states"].shape[0])
            except OSError as error:
                message = str(error).lower()
                if not any(marker in message for marker in ("file signature not found", "bad object header", "bad superblock")):
                    raise
                self.skipped_paths.append(str(path))
                continue
            runs = extract_raw_runs(atomic)
            episode_index = len(self.episodes)
            self.episodes.append({"path": str(path), "length": state_length})
            for run in runs:
                self.samples.append({"path": str(path), "episode_index": episode_index, "run": run})
        if not self.samples:
            roots = ", ".join(str(root) for root in self.roots)
            raise ValueError(f"no stable episodes found below: {roots}")
        self._build_candidate_index()


    def _build_candidate_index(self) -> None:
        """Build cumulative frame counts for lazy run-to-candidate lookup."""
        counts = []
        for sample in self.samples:
            run = sample["run"]
            t0, t1 = int(run["t0"]), int(run["t1"])
            if t1 <= t0:
                raise ValueError(f"run must be non-empty, got [{t0}, {t1})")
            episode_index = int(sample["episode_index"])
            if episode_index < 0 or episode_index >= len(self.episodes):
                raise ValueError(f"run references invalid episode_index={episode_index}")
            episode_length = int(self.episodes[episode_index]["length"])
            if t0 < 0 or t1 > episode_length:
                raise ValueError(
                    f"run [{t0}, {t1}) is outside episode length {episode_length}"
                )
            counts.append(t1 - t0)
        self._candidate_ends = np.cumsum(np.asarray(counts, dtype=np.int64))

    def _candidate_item(self, index: int) -> tuple[dict[str, Any], int]:
        """Return run metadata and its fixed frame anchor for a flat index."""
        size = len(self)
        index = int(index)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(f"candidate index {index} out of range for {size}")
        run_index = int(np.searchsorted(self._candidate_ends, index, side="right"))
        previous_end = int(self._candidate_ends[run_index - 1]) if run_index else 0
        sample = self.samples[run_index]
        run = sample["run"]
        anchor = int(run["t0"]) + index - previous_end
        return sample, anchor


    def _paths(self) -> list[Path]:
        paths = []
        for root in self.roots:
            root_paths = []
            for path in sorted(root.glob("*/aloha_agilex/data/episode_*.hdf5")):
                parts = [part.lower() for part in path.relative_to(root).parts]
                if any(part.startswith(".") or "partial" in part or "incomplete" in part for part in parts):
                    continue
                root_paths.append(path)
            paths.extend(root_paths)
        return paths

    def __len__(self):
        return int(self._candidate_ends[-1]) if len(self._candidate_ends) else 0

    def __getitem__(self, index: int):
        item, anchor = self._candidate_item(index)
        run = item["run"]
        path = Path(item["path"])
        with _open_live_hdf5(path) as h:
            total = int(h["state/ee_states"].shape[0])
            t0, t1 = int(run["t0"]), min(int(run["t1"]), total)
            action_offsets = np.arange(self.action_horizon, dtype=np.int64)
            action_idx, real_action_mask = _query(
                anchor, action_offsets, t0, t1, total
            )
            action_mask = np.ones_like(real_action_mask, dtype=np.bool_)
            state_offsets = np.arange(-(self.state_history - 1), 1, dtype=np.int64)
            state_idx, state_mask = _query(anchor, state_offsets, t0, t1, total)
            # The sole future target aligns to the final action position, action[31].
            future_offsets = np.asarray([31], dtype=np.int64)
            future_idx, real_future_mask = _query(
                anchor, future_offsets, t0, t1, total
            )
            future_mask = np.ones_like(real_future_mask, dtype=np.bool_)
            run_endpoint_idx = t1 - 1
            current_labels = _read_atomic_frame(h, anchor)
            future_labels = _read_atomic_frame(h, int(future_idx[0]))
            bbox_l_original = np.asarray(
                current_labels["bbox_l"], dtype=np.float32
            ).copy()
            bbox_r_original = np.asarray(
                current_labels["bbox_r"], dtype=np.float32
            ).copy()
            bbox_l, bbox_r = bbox_l_original.copy(), bbox_r_original.copy()
            if self.bbox_jitter_enabled:
                bbox_l, bbox_r = jitter_current_bboxes(
                    bbox_l,
                    bbox_r,
                    valid_l=bool(current_labels["valid_l"]),
                    valid_r=bool(current_labels["valid_r"]),
                    probability=self.bbox_jitter_probability,
                    scale_range=self.bbox_jitter_scale_range,
                    center_shift_fraction=(
                        self.bbox_jitter_center_shift_fraction
                    ),
                    min_iou=self.bbox_jitter_min_iou,
                    max_attempts=self.bbox_jitter_max_attempts,
                )

            action_start, action_stop = int(action_idx[0]), int(action_idx[-1]) + 1
            actions = np.asarray(
                h["action/joint_states"][action_start:action_stop], dtype=np.float32
            )[action_idx - action_start]
            # HDF5 fancy indexing requires strictly increasing indices. Read
            # each needed state once, then restore order and terminal repeats.
            state_indices = np.concatenate([state_idx, future_idx, [run_endpoint_idx]])
            unique_states, state_inverse = np.unique(state_indices, return_inverse=True)
            states = np.asarray(
                h["state/ee_states"][unique_states], dtype=np.float32
            )[state_inverse]

            out = {
                "action_sequence": torch.from_numpy(actions),
                "state": torch.from_numpy(states[:self.state_history]),
                "future_state": torch.from_numpy(states[self.state_history:-1]),
                "run_endpoint_state": torch.from_numpy(states[-1]),
                "action_mask": torch.from_numpy(action_mask),
                "future_mask": torch.from_numpy(future_mask),
                "future_skill_id": torch.tensor(
                    future_labels["skill_id"], dtype=torch.long
                ),
                "future_participants_id": torch.tensor(
                    [future_labels["participants_id"]], dtype=torch.long
                ),
                "future_style_id": torch.tensor(
                    [future_labels["style_id"]], dtype=torch.long
                ),
                "future_valid_l": torch.tensor(
                    [future_labels["valid_l"]], dtype=torch.float32
                ),
                "future_valid_r": torch.tensor(
                    [future_labels["valid_r"]], dtype=torch.float32
                ),
                "future_kind_l": torch.tensor(
                    [future_labels["kind_l"]], dtype=torch.long
                ),
                "future_kind_r": torch.tensor(
                    [future_labels["kind_r"]], dtype=torch.long
                ),
                "future_bbox_l": torch.from_numpy(
                    np.asarray(future_labels["bbox_l"], dtype=np.float32)[None]
                ),
                "future_bbox_r": torch.from_numpy(
                    np.asarray(future_labels["bbox_r"], dtype=np.float32)[None]
                ),
                "skill_id": torch.tensor(
                    current_labels["skill_id"], dtype=torch.long
                ),
                "participants_id": torch.tensor(
                    current_labels["participants_id"], dtype=torch.long
                ),
                "style_id": torch.tensor(
                    current_labels["style_id"], dtype=torch.long
                ),
                "valid_l": torch.tensor(
                    current_labels["valid_l"], dtype=torch.float32
                ),
                "valid_r": torch.tensor(
                    current_labels["valid_r"], dtype=torch.float32
                ),
                "kind_l": torch.tensor(current_labels["kind_l"], dtype=torch.long),
                "kind_r": torch.tensor(current_labels["kind_r"], dtype=torch.long),
                "bbox_l": torch.from_numpy(bbox_l),
                "bbox_r": torch.from_numpy(bbox_r),
            }
            for camera in self.visual_cameras:
                frame = _read_camera_frame(h, camera, anchor)
                expected_shape = (self.image_size[1], self.image_size[0], 3)
                if frame.shape != expected_shape:
                    raise ValueError(f"{path}: decoded image must have shape {expected_shape}; got {frame.shape}")
                # DINOv3 consumes float32 CHW ImageNet-normalized tensors.
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
                frame = np.transpose(frame, (2, 0, 1))
                key = "pixel_values" if camera == self.visual_cameras[0] else f"pixel_values_{camera}"
                out[key] = torch.from_numpy(frame).float()
                if camera == self.visual_cameras[0]:
                    future_frames = []
                    for future_index in future_idx:
                        future_frame = _read_camera_frame(h, camera, int(future_index))
                        if future_frame.shape != expected_shape:
                            raise ValueError(f"{path}: decoded future image must have shape {expected_shape}; got {future_frame.shape}")
                        future_frame = future_frame.astype(np.float32) / 255.0
                        future_frame = (
                            future_frame - np.asarray(
                                [0.485, 0.456, 0.406], dtype=np.float32
                            )
                        ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
                        future_frames.append(np.transpose(future_frame, (2, 0, 1)))
                    out["future_pixel_values"] = torch.from_numpy(
                        np.stack(future_frames)
                    ).float()
                    current_mask = _read_camera_mask(h, camera, anchor)
                    if current_mask.shape != (6, *frame.shape[1:]):
                        raise ValueError(f"{path}: decoded mask shape {current_mask.shape} does not match image")
                    out["visual_mask"] = torch.from_numpy(current_mask.copy())
                    out["visual_mask_valid"] = torch.tensor(True, dtype=torch.bool)
            return out


__all__ = [
    "RLAHDF5Dataset", "extract_raw_runs",
    "_decode_v4_mask",
]
