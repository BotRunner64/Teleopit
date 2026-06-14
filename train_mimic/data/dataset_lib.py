#!/usr/bin/env python3
"""Shared utilities for motion dataset build and runtime loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

REQUIRED_NPZ_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
]

NUM_ACTIONS = 29
MOTION_ARRAY_KEYS = [
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
]
HDF5_DATASET_VERSION = 1
DEFAULT_HDF5_MAX_WINDOW_FRAMES = 512
DEFAULT_HDF5_WINDOW_OVERLAP_FRAMES = 64


@dataclass(frozen=True)
class NpzMeta:
    fps: int
    num_frames: int
    num_bodies: int


def parse_window_steps(raw: object | None) -> tuple[int, ...]:
    """Parse future/history window steps in the runtime-compatible order.

    Accepted format mirrors realtime ``reference_steps``:
    ``[0, ...future/non-negative, ...history/negative]``.
    """
    if raw is None:
        return (0,)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"window_steps must be a sequence of ints, got {raw!r}")

    steps: list[int] = []
    for value in raw:
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, int):
            raise ValueError(f"window_steps entries must be ints, got {value!r}")
        steps.append(int(value))
    if not steps:
        raise ValueError("window_steps must contain at least one step")
    if 0 not in steps:
        raise ValueError(f"window_steps must contain 0, got {steps}")

    zero_idx = steps.index(0)
    if zero_idx != 0:
        raise ValueError(
            "window_steps format must be [0, ...future/non-negative, ...history/negative]. "
            f"Got {steps}."
        )
    seen_history = False
    for step in steps[1:]:
        if step < 0:
            seen_history = True
        elif seen_history:
            raise ValueError(
                "window_steps format must be [0, ...future/non-negative, ...history/negative]. "
                f"Got {steps}."
            )
    return tuple(steps)


def compute_clip_sample_ranges(
    clip_lengths: np.ndarray,
    *,
    window_steps: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-clip valid center-frame ranges as ``[start, end)``."""
    lengths = np.asarray(clip_lengths, dtype=np.int64)
    if lengths.ndim != 1:
        raise ValueError(f"clip_lengths must be 1-D, got {lengths.shape}")
    if np.any(lengths <= 0):
        raise ValueError(f"clip_lengths must be > 0, got {lengths.tolist()}")

    steps = parse_window_steps(window_steps)
    max_future = max((step for step in steps if step > 0), default=0)
    min_history = min((step for step in steps if step < 0), default=0)
    starts = np.full_like(lengths, -min_history, dtype=np.int64)
    # Motion interpolation wraps at ``(clip_len - 1) / fps``. Keep the sampled
    # center time strictly below that boundary, and below any requested future step.
    ends = lengths - 1 - max_future
    # A single-frame clip still has one valid current-only sample at frame 0.
    if max_future == 0 and min_history == 0:
        ends = np.where(lengths == 1, 1, ends)
    invalid = np.where(ends <= starts)[0]
    if invalid.size > 0:
        bad_idx = int(invalid[0])
        raise ValueError(
            "window_steps leave no valid center frames for clip "
            f"{bad_idx}: length={int(lengths[bad_idx])}, steps={list(steps)}"
        )
    return starts, ends


def inspect_clip_dict(payload: Mapping[str, Any]) -> NpzMeta:
    keys = set(payload.keys())
    missing = [k for k in REQUIRED_NPZ_KEYS if k not in keys]
    if missing:
        raise ValueError(f"missing NPZ keys: {missing}")

    fps = int(payload["fps"])
    if fps <= 0:
        raise ValueError(f"invalid fps={fps}")

    joint_pos = np.asarray(payload["joint_pos"])
    joint_vel = np.asarray(payload["joint_vel"])
    body_pos_w = np.asarray(payload["body_pos_w"])
    body_quat_w = np.asarray(payload["body_quat_w"])
    body_lin_vel_w = np.asarray(payload["body_lin_vel_w"])
    body_ang_vel_w = np.asarray(payload["body_ang_vel_w"])
    body_names = np.asarray(payload["body_names"])

    if joint_pos.ndim != 2 or joint_pos.shape[1] != NUM_ACTIONS:
        raise ValueError(f"joint_pos must be (T,{NUM_ACTIONS}), got {joint_pos.shape}")
    if joint_vel.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError(f"joint_vel shape mismatch: {joint_vel.shape} vs {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
        raise ValueError(f"body_pos_w must be (T,nb,3), got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[2] != 4:
        raise ValueError(f"body_quat_w must be (T,nb,4), got {body_quat_w.shape}")
    if body_lin_vel_w.ndim != 3 or body_lin_vel_w.shape[2] != 3:
        raise ValueError(f"body_lin_vel_w must be (T,nb,3), got {body_lin_vel_w.shape}")
    if body_ang_vel_w.ndim != 3 or body_ang_vel_w.shape[2] != 3:
        raise ValueError(f"body_ang_vel_w must be (T,nb,3), got {body_ang_vel_w.shape}")

    t = joint_pos.shape[0]
    nb = body_pos_w.shape[1]
    if t <= 0 or nb <= 0:
        raise ValueError("empty time/body dimension")

    for name, arr in [
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("body_pos_w", body_pos_w),
        ("body_quat_w", body_quat_w),
        ("body_lin_vel_w", body_lin_vel_w),
        ("body_ang_vel_w", body_ang_vel_w),
    ]:
        if arr.shape[0] != t:
            raise ValueError(f"{name} first dim mismatch: expected {t}, got {arr.shape[0]}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN/Inf")

    if body_quat_w.shape[1] != nb or body_lin_vel_w.shape[1] != nb or body_ang_vel_w.shape[1] != nb:
        raise ValueError("body array num_bodies mismatch")
    if body_names.ndim != 1 or body_names.shape[0] != nb:
        raise ValueError(f"body_names must be (nb,), got {body_names.shape}")

    quat_norm = np.linalg.norm(body_quat_w, axis=-1)
    if np.min(quat_norm) < 1e-6:
        raise ValueError("body_quat_w contains near-zero norms")

    return NpzMeta(fps=fps, num_frames=t, num_bodies=nb)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def inspect_npz(path: Path) -> NpzMeta:
    if not path.is_file():
        raise FileNotFoundError(f"npz not found: {path}")
    data = np.load(path, allow_pickle=True)
    return inspect_clip_dict({key: data[key] for key in data.files})


def hash_split(clip_id: str, val_percent: int, salt: str = "") -> str:
    payload = f"{salt}:{clip_id}".encode("utf-8")
    bucket = int(hashlib.md5(payload).hexdigest(), 16) % 100
    return "val" if bucket < val_percent else "train"


def resample_along_time(arr: np.ndarray, new_t: int) -> np.ndarray:
    """Resample an array along time axis 0 using linear interpolation."""
    old_t = int(arr.shape[0])
    if old_t == new_t:
        return arr
    if old_t <= 0:
        raise ValueError("cannot resample empty array")
    if new_t <= 0:
        raise ValueError("resampled length must be > 0")
    if old_t == 1:
        return np.repeat(arr, new_t, axis=0)

    pos = np.linspace(0.0, float(old_t - 1), new_t, dtype=np.float64)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, old_t - 1)
    w = (pos - i0).astype(np.float32).reshape((new_t,) + (1,) * (arr.ndim - 1))
    out = arr[i0] * (1.0 - w) + arr[i1] * w
    return out.astype(arr.dtype, copy=False)


def merge_npz_files(
    npz_files: list[Path],
    output_path: Path,
    *,
    target_fps: int | None = None,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    if not npz_files:
        raise ValueError("no npz files to merge")
    if weights is not None and len(weights) != len(npz_files):
        raise ValueError(
            f"weights length ({len(weights)}) != npz_files length ({len(npz_files)})"
        )

    arrays: dict[str, list[np.ndarray]] = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    fps: int | None = None
    body_names: np.ndarray[Any, Any] | None = None
    per_clip_fps: list[int] = []
    per_clip_weights: list[float] = []

    for i, p in enumerate(npz_files):
        d = np.load(p, allow_pickle=True)
        cur_fps = int(d["fps"])
        if cur_fps <= 0:
            raise ValueError(f"invalid fps in {p}: {cur_fps}")
        cur_body_names = np.asarray(d["body_names"])
        if target_fps is not None and target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")

        joint_pos = np.asarray(d["joint_pos"])
        joint_vel = np.asarray(d["joint_vel"])
        body_pos_w = np.asarray(d["body_pos_w"])
        body_quat_w = np.asarray(d["body_quat_w"])
        body_lin_vel_w = np.asarray(d["body_lin_vel_w"])
        body_ang_vel_w = np.asarray(d["body_ang_vel_w"])

        if target_fps is not None and cur_fps != target_fps:
            old_t = int(joint_pos.shape[0])
            new_t = int(round(old_t * float(target_fps) / float(cur_fps)))
            new_t = max(new_t, 1)

            joint_pos = resample_along_time(joint_pos, new_t)
            joint_vel = resample_along_time(joint_vel, new_t)
            body_pos_w = resample_along_time(body_pos_w, new_t)
            body_quat_w = resample_along_time(body_quat_w, new_t)
            body_lin_vel_w = resample_along_time(body_lin_vel_w, new_t)
            body_ang_vel_w = resample_along_time(body_ang_vel_w, new_t)

            quat_norm = np.linalg.norm(body_quat_w, axis=-1, keepdims=True)
            quat_norm = np.where(quat_norm < 1e-8, 1.0, quat_norm)
            body_quat_w = body_quat_w / quat_norm
            cur_fps = target_fps

        if fps is None:
            fps = cur_fps
            body_names = cur_body_names
        else:
            if cur_fps != fps:
                raise ValueError(f"inconsistent fps in merge: {p} has {cur_fps}, expected {fps}")
            if body_names is None or not np.array_equal(cur_body_names, body_names):
                raise ValueError(f"inconsistent body_names in merge: {p}")

        per_clip_fps.append(cur_fps)
        per_clip_weights.append(weights[i] if weights is not None else 1.0)

        arrays["joint_pos"].append(joint_pos)
        arrays["joint_vel"].append(joint_vel)
        arrays["body_pos_w"].append(body_pos_w)
        arrays["body_quat_w"].append(body_quat_w)
        arrays["body_lin_vel_w"].append(body_lin_vel_w)
        arrays["body_ang_vel_w"].append(body_ang_vel_w)

    # Clip boundary metadata
    clip_lengths = np.array(
        [arr.shape[0] for arr in arrays["joint_pos"]], dtype=np.int64
    )
    clip_starts = np.zeros(len(clip_lengths), dtype=np.int64)
    if len(clip_lengths) > 1:
        clip_starts[1:] = np.cumsum(clip_lengths[:-1])

    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    merged["fps"] = int(fps)  # type: ignore[arg-type]
    merged["body_names"] = body_names
    merged["clip_starts"] = clip_starts
    merged["clip_lengths"] = clip_lengths
    merged["clip_fps"] = np.array(per_clip_fps, dtype=np.int64)
    merged["clip_weights"] = np.array(per_clip_weights, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **merged)

    total_frames = int(merged["joint_pos"].shape[0])
    return {
        "output": str(output_path),
        "clips": len(npz_files),
        "num_clips": len(npz_files),
        "frames": total_frames,
        "fps": int(merged["fps"]),
        "duration_s": float(total_frames / max(int(merged["fps"]), 1)),
    }


def merge_clip_dicts(
    clip_dicts: list[dict[str, Any]],
    output_path: Path,
    *,
    target_fps: int | None = None,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Merge a list of in-memory clip array dicts into a single NPZ."""
    merged = merge_clip_dicts_payload(
        clip_dicts,
        target_fps=target_fps,
        weights=weights,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **merged)

    total_frames = int(merged["joint_pos"].shape[0])
    return {
        "output": str(output_path),
        "clips": len(clip_dicts),
        "num_clips": len(clip_dicts),
        "frames": total_frames,
        "fps": int(merged["fps"]),
        "duration_s": float(total_frames / max(int(merged["fps"]), 1)),
    }


def merge_clip_dicts_payload(
    clip_dicts: list[dict[str, Any]],
    *,
    target_fps: int | None = None,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    """Merge in-memory clip array dicts and return a flat motion payload.

    Each dict must have keys: fps, joint_pos, joint_vel, body_pos_w,
    body_quat_w, body_lin_vel_w, body_ang_vel_w, body_names.
    """
    if not clip_dicts:
        raise ValueError("no clip dicts to merge")
    if weights is not None and len(weights) != len(clip_dicts):
        raise ValueError(
            f"weights length ({len(weights)}) != clip_dicts length ({len(clip_dicts)})"
        )
    if target_fps is not None and target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    array_keys = [
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    ]
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in array_keys}
    fps: int | None = None
    body_names: np.ndarray | None = None
    per_clip_fps: list[int] = []
    per_clip_weights: list[float] = []

    for i, cd in enumerate(clip_dicts):
        cur_fps = int(cd["fps"])
        cur_body_names = np.asarray(cd["body_names"])

        clip_arrays = {k: np.asarray(cd[k]) for k in array_keys}

        if target_fps is not None and cur_fps != target_fps:
            old_t = clip_arrays["joint_pos"].shape[0]
            new_t = max(1, round(old_t * target_fps / cur_fps))
            for k in array_keys:
                clip_arrays[k] = resample_along_time(clip_arrays[k], new_t)
            qn = np.linalg.norm(clip_arrays["body_quat_w"], axis=-1, keepdims=True)
            clip_arrays["body_quat_w"] = clip_arrays["body_quat_w"] / np.where(qn < 1e-8, 1.0, qn)
            cur_fps = target_fps

        if fps is None:
            fps = cur_fps
            body_names = cur_body_names
        elif cur_fps != fps:
            raise ValueError(f"inconsistent fps: clip {i} has {cur_fps}, expected {fps}")
        elif body_names is None or not np.array_equal(cur_body_names, body_names):
            raise ValueError(f"inconsistent body_names: clip {i}")

        per_clip_fps.append(cur_fps)
        per_clip_weights.append(weights[i] if weights is not None else 1.0)
        for k in array_keys:
            arrays[k].append(clip_arrays[k])

    clip_lengths = np.array([a.shape[0] for a in arrays["joint_pos"]], dtype=np.int64)
    clip_starts = np.zeros(len(clip_lengths), dtype=np.int64)
    if len(clip_lengths) > 1:
        clip_starts[1:] = np.cumsum(clip_lengths[:-1])

    merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    merged["fps"] = int(fps)  # type: ignore[arg-type]
    merged["body_names"] = body_names
    merged["clip_starts"] = clip_starts
    merged["clip_lengths"] = clip_lengths
    merged["clip_fps"] = np.array(per_clip_fps, dtype=np.int64)
    merged["clip_weights"] = np.array(per_clip_weights, dtype=np.float64)
    return merged


def _window_clip_ranges(
    *,
    clip_start: int,
    clip_length: int,
    max_window_frames: int,
    overlap_frames: int,
) -> list[tuple[int, int]]:
    if clip_length <= 0:
        raise ValueError(f"clip_length must be > 0, got {clip_length}")
    if max_window_frames <= 1:
        raise ValueError(f"max_window_frames must be > 1, got {max_window_frames}")
    if overlap_frames < 0 or overlap_frames >= max_window_frames:
        raise ValueError(
            "overlap_frames must be in [0, max_window_frames), got "
            f"{overlap_frames} for max_window_frames={max_window_frames}"
        )
    if clip_length <= max_window_frames:
        return [(clip_start, clip_length)]

    stride = max_window_frames - overlap_frames
    starts = list(range(0, max(clip_length - max_window_frames + 1, 1), stride))
    tail_start = clip_length - max_window_frames
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return [(clip_start + int(start), max_window_frames) for start in starts]


def write_hdf5_motion_shard(
    merged: Mapping[str, Any],
    output_path: Path,
    *,
    max_window_frames: int = DEFAULT_HDF5_MAX_WINDOW_FRAMES,
    overlap_frames: int = DEFAULT_HDF5_WINDOW_OVERLAP_FRAMES,
) -> dict[str, Any]:
    """Write a merged motion payload as one HDF5 shard with bounded windows.

    The frame arrays remain flat in the HDF5 file.  ``clip_starts`` and
    ``clip_lengths`` describe training windows, not necessarily original clips.
    Long clips are split into overlapping windows to bound runtime cache size.
    """
    missing = [key for key in [*MOTION_ARRAY_KEYS, "fps", "body_names", "clip_starts", "clip_lengths", "clip_fps", "clip_weights"] if key not in merged]
    if missing:
        raise ValueError(f"merged payload missing required keys: {missing}")

    fps = int(merged["fps"])
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    body_names = np.asarray(merged["body_names"]).astype(str)
    original_starts = np.asarray(merged["clip_starts"], dtype=np.int64)
    original_lengths = np.asarray(merged["clip_lengths"], dtype=np.int64)
    original_fps = np.asarray(merged["clip_fps"], dtype=np.int64)
    original_weights = np.asarray(merged["clip_weights"], dtype=np.float64)

    window_starts: list[int] = []
    window_lengths: list[int] = []
    window_fps: list[int] = []
    window_weights: list[float] = []
    source_clip_ids: list[int] = []
    source_start_frames: list[int] = []
    for source_idx, (clip_start, clip_length) in enumerate(zip(original_starts, original_lengths)):
        ranges = _window_clip_ranges(
            clip_start=int(clip_start),
            clip_length=int(clip_length),
            max_window_frames=max_window_frames,
            overlap_frames=overlap_frames,
        )
        per_window_weight = float(original_weights[source_idx]) / float(len(ranges))
        for start, length in ranges:
            window_starts.append(int(start))
            window_lengths.append(int(length))
            window_fps.append(int(original_fps[source_idx]))
            window_weights.append(per_window_weight)
            source_clip_ids.append(int(source_idx))
            source_start_frames.append(int(start - int(clip_start)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "teleopit_motion_hdf5"
        h5.attrs["version"] = HDF5_DATASET_VERSION
        h5.attrs["fps"] = fps
        h5.attrs["max_window_frames"] = int(max_window_frames)
        h5.attrs["overlap_frames"] = int(overlap_frames)
        h5.create_dataset("body_names", data=body_names.astype(object), dtype=str_dt)
        for key in MOTION_ARRAY_KEYS:
            arr = np.asarray(merged[key], dtype=np.float32)
            h5.create_dataset(key, data=arr, chunks=True)
        h5.create_dataset("clip_starts", data=np.asarray(window_starts, dtype=np.int64))
        h5.create_dataset("clip_lengths", data=np.asarray(window_lengths, dtype=np.int64))
        h5.create_dataset("clip_fps", data=np.asarray(window_fps, dtype=np.int64))
        h5.create_dataset("clip_weights", data=np.asarray(window_weights, dtype=np.float64))
        h5.create_dataset("source_clip_ids", data=np.asarray(source_clip_ids, dtype=np.int64))
        h5.create_dataset("source_start_frames", data=np.asarray(source_start_frames, dtype=np.int64))
        h5.create_dataset("source_clip_starts", data=original_starts.astype(np.int64))
        h5.create_dataset("source_clip_lengths", data=original_lengths.astype(np.int64))
        h5.create_dataset("source_clip_fps", data=original_fps.astype(np.int64))
        h5.create_dataset("source_clip_weights", data=original_weights.astype(np.float64))

    total_frames = int(np.asarray(merged["joint_pos"]).shape[0])
    return {
        "path": str(output_path),
        "clips": len(window_lengths),
        "num_clips": len(window_lengths),
        "source_clips": len(original_lengths),
        "frames": total_frames,
        "fps": fps,
        "duration_s": float(total_frames / max(fps, 1)),
        "clip_lengths": [int(v) for v in window_lengths],
        "source_clip_lengths": [int(v) for v in original_lengths],
    }


def read_hdf5_body_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as h5:
        return [
            str(name.decode("utf-8") if isinstance(name, bytes) else name)
            for name in h5["body_names"][()]
        ]


def read_motion_clip(path: Path, clip_index: int) -> dict[str, Any]:
    """Read one source clip from a current HDF5 motion shard path.

    HDF5 shards use source-clip metadata, so ``clip_index`` indexes original
    clips, not bounded training windows.
    """
    if path.suffix == ".h5":
        return read_hdf5_source_clip(path, clip_index)
    raise ValueError(
        f"review/rebuild input must be a current HDF5 shard (.h5), got: {path}"
    )


def read_hdf5_source_clip(path: Path, clip_index: int) -> dict[str, Any]:
    if clip_index < 0:
        raise ValueError(f"HDF5 shard rows require clip_index >= 0: {path}")
    with h5py.File(path, "r") as h5:
        required = [
            "source_clip_starts",
            "source_clip_lengths",
            "source_clip_fps",
            "body_names",
            *MOTION_ARRAY_KEYS,
        ]
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(
                f"HDF5 shard {path} is missing source clip metadata {missing}. "
                "Rebuild the dataset with the current HDF5 writer."
            )
        starts = np.asarray(h5["source_clip_starts"], dtype=np.int64)
        lengths = np.asarray(h5["source_clip_lengths"], dtype=np.int64)
        fps = np.asarray(h5["source_clip_fps"], dtype=np.int64)
        if clip_index >= len(starts):
            raise IndexError(
                f"clip_index {clip_index} out of range [0, {len(starts)}) in {path}"
            )
        start = int(starts[clip_index])
        length = int(lengths[clip_index])
        sl = slice(start, start + length)
        return {
            "fps": int(fps[clip_index]),
            "joint_pos": np.asarray(h5["joint_pos"][sl], dtype=np.float32),
            "joint_vel": np.asarray(h5["joint_vel"][sl], dtype=np.float32),
            "body_pos_w": np.asarray(h5["body_pos_w"][sl], dtype=np.float32),
            "body_quat_w": np.asarray(h5["body_quat_w"][sl], dtype=np.float32),
            "body_lin_vel_w": np.asarray(h5["body_lin_vel_w"][sl], dtype=np.float32),
            "body_ang_vel_w": np.asarray(h5["body_ang_vel_w"][sl], dtype=np.float32),
            "body_names": np.asarray(read_hdf5_body_names(path), dtype=str),
        }


def write_hdf5_manifest(
    split_dir: Path,
    *,
    shard_infos: Sequence[Mapping[str, Any]],
    fps: int,
    body_names: Sequence[str] | np.ndarray,
) -> Path:
    shards = []
    total_windows = 0
    total_frames = 0
    expected_body_names = [str(name) for name in np.asarray(body_names).tolist()]
    for info in shard_infos:
        path = Path(str(info["path"]))
        shard_path = path if path.is_absolute() else split_dir / path
        if shard_path.is_file():
            actual_body_names = read_hdf5_body_names(shard_path)
            if actual_body_names != expected_body_names:
                raise ValueError(
                    f"HDF5 shard body_names mismatch for {shard_path}: "
                    "all shards in a split must use the same body order"
                )
        if path.is_absolute():
            rel_path = path.name if path.parent == split_dir else str(path.relative_to(split_dir))
        else:
            rel_path = str(path)
        clips = int(info.get("clips", info.get("num_clips", 0)))
        frames = int(info.get("frames", 0))
        total_windows += clips
        total_frames += frames
        shards.append({
            "path": rel_path,
            "clips": clips,
            "frames": frames,
        })
    manifest = {
        "format": "teleopit_motion_hdf5",
        "version": HDF5_DATASET_VERSION,
        "fps": int(fps),
        "body_names": expected_body_names,
        "shards": shards,
        "clips": total_windows,
        "frames": total_frames,
    }
    path = split_dir / "manifest.json"
    write_json(path, manifest)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
