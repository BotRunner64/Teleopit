#!/usr/bin/env python3
"""Repartition a shard directory into smaller shard NPZ files.

Each shard is a self-contained merged NPZ (with clip_starts, clip_lengths,
clip_fps, clip_weights, etc.) that can be independently loaded.  Splits are
made at clip boundaries — no clip is ever truncated across shards.

Usage:
    python train_mimic/scripts/data/split_shards.py \
        --input data/datasets/seed_v1/train \
        --output data/datasets/seed_v1/train_small_shards \
        --max_size_gb 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MOTION_ARRAY_KEYS = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def _estimate_frames_per_gb(data: dict) -> float:
    """Estimate how many frames fit in 1 GB based on array dtypes/shapes."""
    one_frame_bytes = 0
    for k in MOTION_ARRAY_KEYS:
        arr = data[k]
        one_frame_bytes += int(np.prod(arr.shape[1:])) * arr.dtype.itemsize
    return 1e9 / max(one_frame_bytes, 1)


def _load_shard_dir(input_dir: Path) -> dict:
    shard_files = sorted(input_dir.glob("*.npz"))
    if not shard_files:
        raise FileNotFoundError(f"no shard NPZ files found in {input_dir}")

    arrays: dict[str, list[np.ndarray]] = {k: [] for k in MOTION_ARRAY_KEYS}
    clip_lengths: list[np.ndarray] = []
    clip_fps: list[np.ndarray] = []
    clip_weights: list[np.ndarray] = []
    clip_sample_starts: list[np.ndarray] = []
    clip_sample_ends: list[np.ndarray] = []
    window_steps: np.ndarray | None = None
    fps: int | None = None
    body_names: np.ndarray | None = None

    for shard_path in shard_files:
        data = np.load(shard_path, allow_pickle=True)
        for key in MOTION_ARRAY_KEYS:
            arrays[key].append(np.asarray(data[key]))
        if fps is None:
            fps = int(data["fps"])
            body_names = np.asarray(data["body_names"])
        else:
            if int(data["fps"]) != fps:
                raise ValueError(
                    f"inconsistent fps across shards: {shard_path} has {int(data['fps'])}, expected {fps}"
                )
            if not np.array_equal(np.asarray(data["body_names"]), body_names):
                raise ValueError(f"inconsistent body_names across shards: {shard_path}")
        clip_lengths.append(np.asarray(data["clip_lengths"]))
        clip_fps.append(np.asarray(data["clip_fps"]))
        clip_weights.append(np.asarray(data["clip_weights"]))
        if "clip_sample_starts" in data:
            clip_sample_starts.append(np.asarray(data["clip_sample_starts"]))
        if "clip_sample_ends" in data:
            clip_sample_ends.append(np.asarray(data["clip_sample_ends"]))
        if "window_steps" in data and window_steps is None:
            window_steps = np.asarray(data["window_steps"])

    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    merged["fps"] = fps
    merged["body_names"] = body_names
    merged["clip_lengths"] = np.concatenate(clip_lengths)
    merged["clip_fps"] = np.concatenate(clip_fps)
    merged["clip_weights"] = np.concatenate(clip_weights)
    merged["clip_starts"] = np.zeros(len(merged["clip_lengths"]), dtype=np.int64)
    if len(merged["clip_lengths"]) > 1:
        merged["clip_starts"][1:] = np.cumsum(merged["clip_lengths"][:-1])
    if clip_sample_starts and sum(len(v) for v in clip_sample_starts) == len(merged["clip_lengths"]):
        merged["clip_sample_starts"] = np.concatenate(clip_sample_starts)
    if clip_sample_ends and sum(len(v) for v in clip_sample_ends) == len(merged["clip_lengths"]):
        merged["clip_sample_ends"] = np.concatenate(clip_sample_ends)
    if window_steps is not None:
        merged["window_steps"] = window_steps
    return merged


def split_shards(
    input_path: Path,
    output_dir: Path,
    max_size_gb: float = 2.0,
) -> list[Path]:
    """Split a shard directory into smaller shards of approximately *max_size_gb* each."""
    print(f"Loading {input_path} ...")
    data = _load_shard_dir(input_path)

    clip_starts = np.asarray(data["clip_starts"])
    clip_lengths = np.asarray(data["clip_lengths"])
    num_clips = len(clip_starts)
    total_frames = int(data["joint_pos"].shape[0])

    has_fps_array = "clip_fps" in data
    has_weights = "clip_weights" in data
    has_window_steps = "window_steps" in data
    has_sample_starts = "clip_sample_starts" in data
    has_sample_ends = "clip_sample_ends" in data
    fps = data["fps"]
    body_names = data["body_names"]

    frames_per_gb = _estimate_frames_per_gb(data)
    max_frames_per_shard = int(frames_per_gb * max_size_gb)

    # --- plan shard boundaries (by clip index) ---
    shard_ranges: list[tuple[int, int]] = []  # (clip_start_idx, clip_end_idx)
    cur_start = 0
    cur_frames = 0
    for i in range(num_clips):
        cl = int(clip_lengths[i])
        if cur_frames + cl > max_frames_per_shard and cur_frames > 0:
            shard_ranges.append((cur_start, i))
            cur_start = i
            cur_frames = 0
        cur_frames += cl
    if cur_start < num_clips:
        shard_ranges.append((cur_start, num_clips))

    print(
        f"  {num_clips} clips, {total_frames} frames -> "
        f"{len(shard_ranges)} shards (target ~{max_size_gb} GB each)"
    )

    # --- write shards ---
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    n_digits = max(3, len(str(len(shard_ranges) - 1)))

    for shard_idx, (c_start, c_end) in enumerate(shard_ranges):
        frame_start = int(clip_starts[c_start])
        if c_end < num_clips:
            frame_end = int(clip_starts[c_end])
        else:
            frame_end = total_frames

        s = slice(frame_start, frame_end)
        shard: dict = {}
        for k in MOTION_ARRAY_KEYS:
            shard[k] = np.asarray(data[k][s])

        # Rebuild clip metadata relative to this shard
        shard_clip_lengths = clip_lengths[c_start:c_end].copy()
        shard_clip_starts = np.zeros(c_end - c_start, dtype=np.int64)
        if len(shard_clip_lengths) > 1:
            shard_clip_starts[1:] = np.cumsum(shard_clip_lengths[:-1])

        shard["fps"] = fps
        shard["body_names"] = body_names
        shard["clip_starts"] = shard_clip_starts
        shard["clip_lengths"] = shard_clip_lengths
        if has_fps_array:
            shard["clip_fps"] = np.asarray(data["clip_fps"][c_start:c_end])
        if has_weights:
            shard["clip_weights"] = np.asarray(data["clip_weights"][c_start:c_end])
        if has_window_steps:
            shard["window_steps"] = np.asarray(data["window_steps"])
        if has_sample_starts:
            shard["clip_sample_starts"] = np.asarray(data["clip_sample_starts"][c_start:c_end])
        if has_sample_ends:
            shard["clip_sample_ends"] = np.asarray(data["clip_sample_ends"][c_start:c_end])

        shard_name = f"shard_{shard_idx:0{n_digits}d}.npz"
        shard_path = output_dir / shard_name
        np.savez(shard_path, **shard)

        shard_frames = frame_end - frame_start
        shard_size_mb = shard_path.stat().st_size / 1e6
        print(
            f"  {shard_name}: {c_end - c_start} clips, "
            f"{shard_frames} frames, {shard_size_mb:.0f} MB"
        )
        shard_paths.append(shard_path)

    print(f"Done. {len(shard_paths)} shards written to {output_dir}")
    return shard_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Repartition a shard directory into smaller shards")
    parser.add_argument("--input", required=True, type=Path, help="Input shard directory")
    parser.add_argument("--output", required=True, type=Path, help="Output shard directory")
    parser.add_argument(
        "--max_size_gb",
        type=float,
        default=2.0,
        help="Target max size per shard in GB (default: 2.0)",
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"Error: shard directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    split_shards(args.input, args.output, args.max_size_gb)


if __name__ == "__main__":
    main()
