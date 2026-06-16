from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

import h5py
import mujoco
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from train_mimic.data.dataset_lib import (
    MOTION_ARRAY_KEYS,
    compute_clip_sample_ranges,
    compute_dataset_stats,
    find_precomputed_motion_shards,
    parse_window_steps,
    validate_precomputed_motion_shard,
)

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
_LOG = logging.getLogger(__name__)


def _batched_quat_slerp(
    q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Batched quaternion slerp.  q0, q1: (..., 4) wxyz, t: broadcastable."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs()

    t = t.unsqueeze(-1) if t.dim() < q0.dim() else t

    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_omega = torch.sin(omega)
    safe = sin_omega.abs() > 1e-8

    w0_slerp = torch.sin((1.0 - t) * omega) / sin_omega
    w1_slerp = torch.sin(t * omega) / sin_omega

    w0 = torch.where(safe, w0_slerp, 1.0 - t)
    w1 = torch.where(safe, w1_slerp, t)

    result = w0 * q0 + w1 * q1
    return result / result.norm(dim=-1, keepdim=True)


@dataclass(frozen=True)
class _Hdf5ClipRef:
    shard_index: int
    start: int
    length: int
    fps: int


@dataclass
class _MotionBatch:
    tensors: dict[str, torch.Tensor]
    frame_offsets: torch.Tensor
    lengths: torch.Tensor
    fps: torch.Tensor
    sample_starts: torch.Tensor
    sample_ends: torch.Tensor
    global_ids: torch.Tensor

    def pin_memory(self) -> "_MotionBatch":
        return _MotionBatch(
            tensors={key: value.pin_memory() for key, value in self.tensors.items()},
            frame_offsets=self.frame_offsets.pin_memory(),
            lengths=self.lengths.pin_memory(),
            fps=self.fps.pin_memory(),
            sample_starts=self.sample_starts.pin_memory(),
            sample_ends=self.sample_ends.pin_memory(),
            global_ids=self.global_ids.pin_memory(),
        )


@dataclass
class _MotionClipSample:
    tensors: dict[str, torch.Tensor]
    length: int
    fps: int
    sample_start: int
    sample_end: int
    global_id: int


class _WeightedInfiniteClipBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        sample_weights: torch.Tensor,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.sample_weights = sample_weights.cpu().to(dtype=torch.float32)
        self.batch_size = int(batch_size)
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            ids = torch.multinomial(
                self.sample_weights,
                self.batch_size,
                replacement=True,
                generator=self._rng,
            )
            yield [int(value) for value in ids.tolist()]


class _Hdf5MotionDataset(Dataset[_MotionClipSample]):
    def __init__(
        self,
        *,
        refs: list[_Hdf5ClipRef],
        shard_paths: list[Path],
        body_idx_np: np.ndarray,
        window_steps: tuple[int, ...],
    ) -> None:
        self.refs = refs
        self._shard_paths = shard_paths
        self.body_idx_np = body_idx_np
        self.window_steps = window_steps
        self._h5_handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> _MotionClipSample:
        ref = self.refs[int(index)]
        sl = slice(ref.start, ref.start + ref.length)
        h5 = self._h5_handle(ref.shard_index)
        joint_pos = np.asarray(h5["joint_pos"][sl], dtype=np.float32)

        sample_starts, sample_ends = compute_clip_sample_ranges(
            np.asarray([ref.length], dtype=np.int64),
            window_steps=self.window_steps,
        )
        tensors = {
            "joint_pos": torch.from_numpy(joint_pos),
            "joint_vel": torch.from_numpy(np.asarray(h5["joint_vel"][sl], dtype=np.float32)),
            "body_pos_w": torch.from_numpy(
                np.asarray(h5["body_pos_w"][sl], dtype=np.float32)[:, self.body_idx_np]
            ),
            "body_quat_w": torch.from_numpy(
                np.asarray(h5["body_quat_w"][sl], dtype=np.float32)[:, self.body_idx_np]
            ),
            "body_lin_vel_w": torch.from_numpy(
                np.asarray(h5["body_lin_vel_w"][sl], dtype=np.float32)[:, self.body_idx_np]
            ),
            "body_ang_vel_w": torch.from_numpy(
                np.asarray(h5["body_ang_vel_w"][sl], dtype=np.float32)[:, self.body_idx_np]
            ),
        }
        return _MotionClipSample(
            tensors=tensors,
            length=ref.length,
            fps=ref.fps,
            sample_start=int(sample_starts[0]),
            sample_end=int(sample_ends[0]),
            global_id=int(index),
        )

    def _h5_handle(self, shard_index: int) -> h5py.File:
        handle = self._h5_handles.get(shard_index)
        if handle is not None and handle.id:
            return handle
        handle = h5py.File(self._shard_paths[shard_index], "r")
        self._h5_handles[shard_index] = handle
        return handle

    def close(self) -> None:
        for handle in self._h5_handles.values():
            if handle.id:
                handle.close()
        self._h5_handles.clear()


def _collate_motion_clips(samples: list[_MotionClipSample]) -> _MotionBatch:
    if not samples:
        raise ValueError("Motion cache DataLoader produced an empty batch")
    arrays: dict[str, torch.Tensor] = {}
    for key in MOTION_ARRAY_KEYS:
        arrays[key] = torch.cat([sample.tensors[key] for sample in samples], dim=0)

    lengths = torch.tensor([sample.length for sample in samples], dtype=torch.long)
    frame_offsets = torch.zeros(len(samples), dtype=torch.long)
    if len(samples) > 1:
        frame_offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

    return _MotionBatch(
        tensors=arrays,
        frame_offsets=frame_offsets,
        lengths=lengths,
        fps=torch.tensor([sample.fps for sample in samples], dtype=torch.float32),
        sample_starts=torch.tensor([sample.sample_start for sample in samples], dtype=torch.long),
        sample_ends=torch.tensor([sample.sample_end for sample in samples], dtype=torch.long),
        global_ids=torch.tensor([sample.global_id for sample in samples], dtype=torch.long),
    )


def _motion_worker_init(worker_id: int) -> None:
    del worker_id
    torch.set_num_threads(1)


class _Hdf5MotionCache:
    def __init__(
        self,
        motion_dir: Path,
        *,
        body_idx_np: np.ndarray,
        device: str,
        window_steps: tuple[int, ...],
        cache_num_clips: int,
        seed: int,
        dataloader_num_workers: int,
        dataloader_prefetch_factor: int,
        dataloader_pin_memory: bool,
    ) -> None:
        if cache_num_clips <= 0:
            raise ValueError(f"cache_num_clips must be positive, got {cache_num_clips}")

        self.motion_dir = motion_dir
        self.body_idx_np = body_idx_np
        self.device = device
        self.window_steps = window_steps
        self.cache_num_clips = int(cache_num_clips)
        self.dataloader_num_workers = max(0, int(dataloader_num_workers))
        self.dataloader_prefetch_factor = max(1, int(dataloader_prefetch_factor))
        self.dataloader_pin_memory = bool(dataloader_pin_memory)
        self._device = torch.device(device)
        self._copy_stream: torch.cuda.Stream | None = None
        self._next_ready_event: torch.cuda.Event | None = None
        self._shard_paths = find_precomputed_motion_shards(motion_dir)
        stats = compute_dataset_stats(motion_dir, precomputed=True)
        self.body_names = np.asarray(stats["body_names"], dtype=str)
        for shard_path in self._shard_paths:
            validate_precomputed_motion_shard(shard_path)
        _LOG.info(
            "Motion dataset: root=%s shards=%d windows=%d source_clips=%d frames=%d fps=%s",
            motion_dir,
            stats["shards"],
            stats["windows"],
            stats["source_clips"],
            stats["frames"],
            stats["fps"],
        )

        max_future = max((step for step in self.window_steps if step > 0), default=0)
        max_history = -min((step for step in self.window_steps if step < 0), default=0)
        min_clip_length = max_history + 1 + max_future + 1  # +1 for interpolation

        refs: list[_Hdf5ClipRef] = []
        skipped_short = 0
        for shard_index, shard_path in enumerate(self._shard_paths):
            with h5py.File(shard_path, "r") as h5:
                starts = np.asarray(h5["clip_starts"], dtype=np.int64)
                lengths = np.asarray(h5["clip_lengths"], dtype=np.int64)
                fps = np.asarray(h5["clip_fps"], dtype=np.int64)
                for start, length, cur_fps in zip(starts, lengths, fps):
                    if int(length) < min_clip_length:
                        skipped_short += 1
                        continue
                    refs.append(_Hdf5ClipRef(
                        shard_index=shard_index,
                        start=int(start),
                        length=int(length),
                        fps=int(cur_fps),
                    ))
        if not refs:
            raise ValueError(f"HDF5 motion dataset is empty: {motion_dir}")
        if skipped_short > 0:
            _LOG.warning(
                "Ignoring %d HDF5 motion windows shorter than %d frames (window_steps=%s)",
                skipped_short,
                min_clip_length,
                list(self.window_steps),
            )
        self.refs = refs
        ref_lengths_np = np.asarray([ref.length for ref in refs], dtype=np.int64)
        ref_starts_np, ref_ends_np = compute_clip_sample_ranges(
            ref_lengths_np,
            window_steps=self.window_steps,
        )
        ref_fps_np = np.asarray([ref.fps for ref in refs], dtype=np.float32)
        ref_valid_seconds = (ref_ends_np - ref_starts_np).astype(np.float32) / ref_fps_np
        if np.any(ref_valid_seconds <= 0.0):
            raise ValueError(
                "HDF5 motion dataset contains windows with no valid sample duration "
                f"after applying window_steps={list(self.window_steps)}"
            )
        self.global_sample_weights = torch.as_tensor(ref_valid_seconds, dtype=torch.float32)
        total_weight = float(self.global_sample_weights.sum().item())
        if total_weight <= 0.0:
            raise ValueError(f"HDF5 motion dataset has no positive sample duration: {motion_dir}")
        self.generation = 0
        self._dataset = _Hdf5MotionDataset(
            refs=self.refs,
            shard_paths=self._shard_paths,
            body_idx_np=self.body_idx_np,
            window_steps=self.window_steps,
        )
        self._sampler = _WeightedInfiniteClipBatchSampler(
            sample_weights=self.global_sample_weights,
            batch_size=self.cache_num_clips,
            seed=seed,
        )
        loader_kwargs: dict[str, object] = {}
        if self.dataloader_num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["worker_init_fn"] = _motion_worker_init
        self._loader = DataLoader(
            self._dataset,
            batch_sampler=self._sampler,
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory and self._device.type == "cuda",
            collate_fn=_collate_motion_clips,
            **loader_kwargs,
        )
        self._iterator = iter(self._loader)
        self.current = self._stage_batch(self._load_next_cpu_batch(), wait=True)
        self._next_batch = self._stage_batch(self._load_next_cpu_batch(), wait=False)

    def _load_next_cpu_batch(self, *, log_wait: bool = False) -> _MotionBatch:
        start = time.perf_counter()
        batch = next(self._iterator)
        elapsed = time.perf_counter() - start
        if log_wait and elapsed > 1e-3:
            _LOG.info(
                "Waited %.3fs for asynchronous HDF5 motion cache DataLoader",
                elapsed,
            )
        return batch

    def _stage_batch(self, batch: _MotionBatch, *, wait: bool) -> _MotionBatch:
        if self._device.type != "cuda":
            tensors = {key: value.to(self._device) for key, value in batch.tensors.items()}
            return _MotionBatch(
                tensors=tensors,
                frame_offsets=batch.frame_offsets.to(self._device),
                lengths=batch.lengths.to(self._device),
                fps=batch.fps.to(self._device),
                sample_starts=batch.sample_starts.to(self._device),
                sample_ends=batch.sample_ends.to(self._device),
                global_ids=batch.global_ids.to(self._device),
            )

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self._device)
        with torch.cuda.stream(self._copy_stream):
            tensors = {
                key: value.to(self._device, non_blocking=True)
                for key, value in batch.tensors.items()
            }
            staged = _MotionBatch(
                tensors=tensors,
                frame_offsets=batch.frame_offsets.to(self._device, non_blocking=True),
                lengths=batch.lengths.to(self._device, non_blocking=True),
                fps=batch.fps.to(self._device, non_blocking=True),
                sample_starts=batch.sample_starts.to(self._device, non_blocking=True),
                sample_ends=batch.sample_ends.to(self._device, non_blocking=True),
                global_ids=batch.global_ids.to(self._device, non_blocking=True),
            )
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        if wait:
            torch.cuda.current_stream(self._device).wait_event(event)
        else:
            self._next_ready_event = event
        return staged

    def _wait_next_ready(self) -> None:
        if self._next_ready_event is None:
            return
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).wait_event(self._next_ready_event)
        self._next_ready_event = None

    def _materialize_batch_by_global_ids(self, global_ids: torch.Tensor) -> _MotionBatch:
        samples = [self._dataset[int(idx)] for idx in global_ids.tolist()]
        batch = _collate_motion_clips(samples)
        return self._stage_batch(batch, wait=True)

    def _sample_global_ids(self) -> torch.Tensor:
        ids = next(iter(self._sampler))
        return torch.tensor(ids, dtype=torch.long)

    def _load_random_batch(self) -> _MotionBatch:
        return self._materialize_batch_by_global_ids(self._sample_global_ids())

    def _load_batch(self, global_ids: torch.Tensor) -> _MotionBatch:
        return self._materialize_batch_by_global_ids(global_ids.cpu().to(dtype=torch.long))

    def advance(self) -> None:
        start = time.perf_counter()
        self._wait_next_ready()
        elapsed = time.perf_counter() - start
        if elapsed > 1e-3:
            _LOG.info(
                "Waited %.3fs for asynchronous HDF5 motion cache staging",
                elapsed,
            )
        self.current = self._next_batch
        self._next_batch = self._stage_batch(
            self._load_next_cpu_batch(log_wait=True),
            wait=False,
        )
        self.generation += 1

    def close(self) -> None:
        self._dataset.close()
        iterator = getattr(self, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MotionLib:
    """Clip-aware motion library.

    Loads a bounded subset of HDF5 motion windows into a GPU-resident cache.
    Sampling and interpolation operate on cache-local clip ids; the next cache
    is staged in memory and swapped at a rollout barrier by ``MotionCommand``.
    """

    def __init__(
        self,
        motion_file: str,
        body_indexes: torch.Tensor,
        body_names: tuple[str, ...] | list[str] | None = None,
        device: str = "cpu",
        window_steps: tuple[int, ...] | list[int] | None = None,
        cache_num_clips: int = 8192,
        cache_seed: int = 0,
        dataloader_num_workers: int = 2,
        dataloader_prefetch_factor: int = 1,
        dataloader_pin_memory: bool = True,
    ) -> None:
        self._device = device
        self.window_steps = parse_window_steps(window_steps)

        motion_path = Path(motion_file)
        if not motion_path.exists():
            raise FileNotFoundError(
                f"motion_file must be a dataset root directory or .h5 shard, got: {motion_file}"
            )
        stats = compute_dataset_stats(motion_path, precomputed=True)

        if body_names is None:
            body_idx_np = body_indexes.cpu().numpy()
        else:
            dataset_body_names = [str(name) for name in stats["body_names"]]
            dataset_body_index_by_name = {
                name: index for index, name in enumerate(dataset_body_names)
            }
            missing_body_names = [
                name for name in body_names if name not in dataset_body_index_by_name
            ]
            if missing_body_names:
                raise ValueError(
                    "Motion dataset body_names do not contain all requested tracking "
                    f"bodies. Missing: {missing_body_names}. "
                    "Rebuild the dataset with the current G1 body metadata or update "
                    "motion command body_names."
                )
            body_idx_np = np.asarray(
                [dataset_body_index_by_name[name] for name in body_names],
                dtype=np.int64,
            )

        self._cache = _Hdf5MotionCache(
            motion_path,
            body_idx_np=body_idx_np,
            device=device,
            window_steps=self.window_steps,
            cache_num_clips=cache_num_clips,
            seed=cache_seed,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_prefetch_factor=dataloader_prefetch_factor,
            dataloader_pin_memory=dataloader_pin_memory,
        )
        self._set_batch(self._cache.current)

    def _set_batch(self, batch: _MotionBatch) -> None:
        self._batch = batch
        self._joint_pos_t = batch.tensors["joint_pos"]
        self._joint_vel_t = batch.tensors["joint_vel"]
        self._body_pos_w_t = batch.tensors["body_pos_w"]
        self._body_quat_w_t = batch.tensors["body_quat_w"]
        self._body_lin_vel_w_t = batch.tensors["body_lin_vel_w"]
        self._body_ang_vel_w_t = batch.tensors["body_ang_vel_w"]
        self.clip_frame_offsets = batch.frame_offsets

        self.clip_lengths = batch.lengths
        self.clip_fps = batch.fps
        self.num_clips = int(batch.lengths.shape[0])
        self.time_step_total = int(batch.lengths.max().item())
        self.clip_dt = 1.0 / self.clip_fps
        self.clip_duration_s = (self.clip_lengths.float() - 1.0) * self.clip_dt
        self.clip_sample_starts = batch.sample_starts
        self.clip_sample_ends = batch.sample_ends
        self.clip_sample_start_s = self.clip_sample_starts.float() * self.clip_dt
        self.clip_sample_end_s = self.clip_sample_ends.float() * self.clip_dt
        # Kept for introspection/logging; these are cache-local flat frame offsets.
        self.clip_starts = self.clip_frame_offsets
        self.generation = self._cache.generation

    def advance_cache(self) -> None:
        self._cache.advance()
        self._set_batch(self._cache.current)

    def close(self) -> None:
        self._cache.close()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        """Sample *n* cache-local clip indices uniformly."""
        return torch.randint(0, self.num_clips, (n,), device=self._device)

    def sample_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Uniform random time over valid center frames for each motion id."""
        sample_starts = self.clip_sample_starts[motion_ids].float()
        sample_ends = self.clip_sample_ends[motion_ids].float()
        valid_lengths = sample_ends - sample_starts
        if torch.any(valid_lengths <= 0):
            raise ValueError(
                "Requested window_steps leave no valid frames for one or more sampled clips. "
                f"window_steps={list(self.window_steps)}"
            )
        frame_f = sample_starts + torch.rand_like(sample_starts) * valid_lengths
        return frame_f / self.clip_fps[motion_ids]

    def sample_start_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Return the earliest valid center time for each motion id."""
        return self.clip_sample_start_s[motion_ids]

    def _compute_interpolation_state(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        steps: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        fps = self.clip_fps[motion_ids]
        lengths = self.clip_lengths[motion_ids]
        durations = self.clip_duration_s[motion_ids]

        safe_dur = torch.clamp(durations, min=1e-6)
        motion_times = torch.fmod(motion_times, safe_dur)
        motion_times = torch.where(motion_times < 0, motion_times + safe_dur, motion_times)

        step_offsets = torch.tensor(steps, dtype=torch.float32, device=self._device)
        frame_f = motion_times[:, None] * fps[:, None] + step_offsets[None, :]
        frame_i0 = frame_f.long()
        frame_i1 = frame_i0 + 1
        alpha = frame_f - frame_i0.float()

        zero = torch.zeros_like(frame_i0)
        max_frame = (lengths - 1)[:, None].expand_as(frame_i0)
        frame_i0 = torch.clamp(frame_i0, min=zero, max=max_frame)
        frame_i1 = torch.clamp(frame_i1, min=zero, max=max_frame)

        window = len(steps)
        return frame_i0.reshape(-1), frame_i1.reshape(-1), alpha.reshape(-1), window

    _ALL_KEYS = frozenset((
        "joint_pos", "joint_vel",
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    ))

    def get_window_frames(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        *,
        window_steps: tuple[int, ...] | list[int] | None = None,
        keys: frozenset[str] | None = None,
        body_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Interpolate motion frames for a batch of (motion_id, time) pairs.

        All indexing, lerp, and slerp run entirely on device.

        Args:
            keys: If given, only compute these arrays (skip the rest).
            body_indices: If given, only index these bodies for body_* arrays.
                This dramatically reduces slerp cost when only the anchor
                body is needed.
        """
        steps = parse_window_steps(self.window_steps if window_steps is None else window_steps)
        idx0, idx1, alpha, window = self._compute_interpolation_state(
            motion_ids,
            motion_times,
            steps,
        )
        batch = motion_ids.shape[0]
        frame_offsets = self.clip_frame_offsets[motion_ids]
        flat_idx0 = (frame_offsets[:, None] + idx0.reshape(batch, window)).reshape(-1)
        flat_idx1 = (frame_offsets[:, None] + idx1.reshape(batch, window)).reshape(-1)
        want = self._ALL_KEYS if keys is None else keys

        result: dict[str, torch.Tensor] = {}

        # joint_pos, joint_vel: (T, D) → GPU gather + lerp
        a1 = alpha[:, None]
        for key, arr_t in (("joint_pos", self._joint_pos_t), ("joint_vel", self._joint_vel_t)):
            if key not in want:
                continue
            v0, v1 = arr_t[flat_idx0], arr_t[flat_idx1]
            result[key] = (v0 + a1 * (v1 - v0)).reshape(batch, window, -1)

        # body arrays: (T, B, D) — GPU gather + lerp, optionally pre-slice bodies
        a2 = alpha[:, None, None]
        for key, arr_t in (
            ("body_pos_w", self._body_pos_w_t),
            ("body_lin_vel_w", self._body_lin_vel_w_t),
            ("body_ang_vel_w", self._body_ang_vel_w_t),
        ):
            if key not in want:
                continue
            if body_indices is not None:
                v0 = arr_t[flat_idx0][:, body_indices]
                v1 = arr_t[flat_idx1][:, body_indices]
            else:
                v0, v1 = arr_t[flat_idx0], arr_t[flat_idx1]
            interp = v0 + a2 * (v1 - v0)
            result[key] = interp.reshape(batch, window, *interp.shape[1:])

        # body_quat_w: GPU slerp, optionally pre-slice bodies
        if "body_quat_w" in want:
            if body_indices is not None:
                q0 = self._body_quat_w_t[flat_idx0][:, body_indices]
                q1 = self._body_quat_w_t[flat_idx1][:, body_indices]
            else:
                q0 = self._body_quat_w_t[flat_idx0]
                q1 = self._body_quat_w_t[flat_idx1]
            nb = q0.shape[1]
            q0_flat = q0.reshape(-1, 4)
            q1_flat = q1.reshape(-1, 4)
            alpha_flat = alpha.unsqueeze(-1).expand(batch * window, nb).reshape(-1)
            result["body_quat_w"] = (
                _batched_quat_slerp(q0_flat, q1_flat, alpha_flat)
                .reshape(batch, window, nb, 4)
            )
        return result

    # ------------------------------------------------------------------
    # Frame interpolation
    # ------------------------------------------------------------------

    def get_frames(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Look up interpolated motion state for ``(motion_id, time)`` pairs.

        Handles time wrapping (loop) and sub-frame linear / slerp interpolation.
        All computation (indexing, lerp, slerp) runs on GPU.
        """
        windowed = self.get_window_frames(motion_ids, motion_times, window_steps=(0,))
        return {
            key: value[:, 0] for key, value in windowed.items()
        }

def _validate_rewind_sampling_cfg(cfg: Any) -> None:
    if cfg.rewind_min_steps < 0:
        raise ValueError(
            f"rewind_min_steps must be non-negative, got {cfg.rewind_min_steps}"
        )
    if cfg.rewind_max_steps < cfg.rewind_min_steps:
        raise ValueError(
            "rewind_max_steps must be >= rewind_min_steps, got "
            f"{cfg.rewind_max_steps} < {cfg.rewind_min_steps}"
        )
    if not 0.0 <= cfg.rewind_prob <= 1.0:
        raise ValueError(f"rewind_prob must be in [0, 1], got {cfg.rewind_prob}")


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        if self.cfg.sampling_mode == "rewind":
            _validate_rewind_sampling_cfg(self.cfg)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long,
            device=self.device,
        )

        self.motion = MotionLib(
            self.cfg.motion_file,
            self.body_indexes,
            body_names=self.cfg.body_names,
            device=self.device,
            window_steps=self.cfg.window_steps,
            cache_num_clips=self.cfg.cache_num_clips,
            cache_seed=self.cfg.cache_seed,
            dataloader_num_workers=self.cfg.cache_dataloader_num_workers,
            dataloader_prefetch_factor=self.cfg.cache_dataloader_prefetch_factor,
            dataloader_pin_memory=self.cfg.cache_dataloader_pin_memory,
        )
        self._motion_cache_step_counter = 0
        self._motion_cache_swap_pending = False

        # Per-env motion state: clip id + elapsed time (seconds)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._step_dt = env.step_dt

        # Cached interpolated frames — refreshed every step
        self._cached_frames: dict[str, torch.Tensor] = {}
        self._refresh_frame_cache()

        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[:, :, 0] = 1.0

        nb = len(cfg.body_names)
        self.body_pos_b = torch.zeros(self.num_envs, nb, 3, device=self.device)
        self.body_quat_b = torch.zeros(self.num_envs, nb, 4, device=self.device)
        self.body_quat_b[:, :, 0] = 1.0
        self.body_lin_vel_b = torch.zeros(self.num_envs, nb, 3, device=self.device)
        self.body_ang_vel_b = torch.zeros(self.num_envs, nb, 3, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_anchor_ang_vel"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.feet_body_names:
            self._feet_body_indexes = [
                self.cfg.body_names.index(n) for n in self.cfg.feet_body_names
            ]
        else:
            self._feet_body_indexes = []
        self.feet_standing = torch.zeros(
            (self.num_envs, max(len(self._feet_body_indexes), 1)),
            dtype=torch.float32,
            device=self.device,
        )
        if self._feet_body_indexes:
            self._update_feet_standing()

        # Ghost model created lazily on first visualization
        self._ghost_model: mujoco.MjModel | None = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # ------------------------------------------------------------------
    # Frame cache
    # ------------------------------------------------------------------

    # Keys needed by window_* properties (ref_window_b observation).
    _WINDOW_KEYS = frozenset(("joint_pos", "joint_vel", "body_quat_w"))

    def _refresh_frame_cache(self) -> None:
        self._cached_frames = self.motion.get_frames(self.motion_ids, self.motion_times)
        if self.cfg.window_steps != (0,):
            self._cached_window_frames = self.motion.get_window_frames(
                self.motion_ids, self.motion_times,
                keys=self._WINDOW_KEYS,
                body_indices=[self.motion_anchor_body_index],
            )
        else:
            self._cached_window_frames = {}

    def _update_feet_standing(self) -> None:
        """Compute feet contact state from reference motion (z + velocity thresholds)."""
        if not self._feet_body_indexes:
            return
        feet_pos_w = self.body_pos_w[:, self._feet_body_indexes]
        feet_vel_w = self.body_lin_vel_w[:, self._feet_body_indexes]
        root_vxy = torch.norm(
            self.body_lin_vel_w[:, 0, :2], dim=-1, keepdim=True
        ).clamp_min(1.0)
        feet_vxy = torch.norm(feet_vel_w[..., :2], dim=-1)
        feet_vz = feet_vel_w[..., 2].abs()
        feet_z = feet_pos_w[..., 2]
        standing = (
            (feet_z < self.cfg.feet_standing_z_threshold)
            & (feet_vxy < self.cfg.feet_standing_vxy_threshold * root_vxy)
            & (feet_vz < self.cfg.feet_standing_vz_threshold * root_vxy)
        )
        self.feet_standing = standing.float()

    # ------------------------------------------------------------------
    # Properties — motion reference (from cached interpolated frames)
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._cached_frames["joint_pos"]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._cached_frames["joint_vel"]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._cached_frames["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._cached_frames["body_quat_w"]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_ang_vel_w"]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return (
            self._cached_frames["body_pos_w"][:, self.motion_anchor_body_index]
            + self._env.scene.env_origins
        )

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._cached_frames["body_quat_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_lin_vel_w"][:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._cached_frames["body_ang_vel_w"][:, self.motion_anchor_body_index]

    # ------------------------------------------------------------------
    # Properties — windowed reference (from cached window frames)
    # ------------------------------------------------------------------

    @property
    def window_anchor_quat_w(self) -> torch.Tensor:
        """(B, W, 4) - ref anchor quaternion at each window step."""
        # Window frames are fetched with body_indices=[anchor], so body dim is 0.
        return self._cached_window_frames["body_quat_w"][:, :, 0]

    @property
    def window_joint_pos(self) -> torch.Tensor:
        """(B, W, J) - ref joint positions at each window step."""
        return self._cached_window_frames["joint_pos"]

    @property
    def window_joint_vel(self) -> torch.Tensor:
        """(B, W, J) - ref joint velocities at each window step."""
        return self._cached_window_frames["joint_vel"]

    # ------------------------------------------------------------------
    # Properties — robot actual state (unchanged)
    # ------------------------------------------------------------------

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

    # ------------------------------------------------------------------
    # Properties — velocity-driven commands
    # ------------------------------------------------------------------

    @property
    def anchor_lin_vel_heading(self) -> torch.Tensor:
        """Reference anchor linear velocity in its heading (yaw-only) frame."""
        return quat_apply(quat_inv(yaw_quat(self.anchor_quat_w)), self.anchor_lin_vel_w)

    @property
    def anchor_yaw_rate(self) -> torch.Tensor:
        """Reference anchor yaw angular velocity (world-z component)."""
        return self.anchor_ang_vel_w[:, 2]

    @property
    def robot_anchor_lin_vel_heading(self) -> torch.Tensor:
        """Robot anchor linear velocity in its own heading (yaw-only) frame."""
        return quat_apply(
            quat_inv(yaw_quat(self.robot_anchor_quat_w)), self.robot_anchor_lin_vel_w
        )

    @property
    def robot_anchor_yaw_rate(self) -> torch.Tensor:
        """Robot anchor yaw angular velocity (world-z component)."""
        return self.robot_anchor_ang_vel_w[:, 2]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(
            self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
        )
        self.metrics["error_anchor_rot"] = quat_error_magnitude(
            self.anchor_quat_w, self.robot_anchor_quat_w
        )
        self.metrics["error_anchor_lin_vel"] = torch.norm(
            self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
        )
        self.metrics["error_anchor_ang_vel"] = torch.norm(
            self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
        )

        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(
            self.joint_pos - self.robot_joint_pos, dim=-1
        )
        self.metrics["error_joint_vel"] = torch.norm(
            self.joint_vel - self.robot_joint_vel, dim=-1
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _uniform_sampling(self, env_ids: torch.Tensor):
        self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
        self.motion_times[env_ids] = self.motion.sample_times(self.motion_ids[env_ids])

    def _rewind_sampling(self, env_ids: torch.Tensor) -> None:
        _validate_rewind_sampling_cfg(self.cfg)

        previous_motion_ids = self.motion_ids[env_ids].clone()
        previous_motion_times = self.motion_times[env_ids].clone()
        self._uniform_sampling(env_ids)
        if env_ids.numel() == 0 or self.cfg.rewind_prob <= 0.0:
            return

        episode_failed = self._env.termination_manager.terminated[env_ids]
        use_rewind = episode_failed & (
            torch.rand(env_ids.numel(), device=self.device) < self.cfg.rewind_prob
        )
        if not torch.any(use_rewind):
            return

        rewind_env_ids = env_ids[use_rewind]
        rewind_steps = torch.randint(
            self.cfg.rewind_min_steps,
            self.cfg.rewind_max_steps + 1,
            (rewind_env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )
        rewind_s = rewind_steps.to(dtype=self.motion_times.dtype) * float(self._step_dt)
        motion_ids = previous_motion_ids[use_rewind]
        rewind_times = previous_motion_times[use_rewind] - rewind_s
        rewind_times = torch.maximum(
            rewind_times,
            self.motion.clip_sample_start_s[motion_ids],
        )

        self.motion_ids[rewind_env_ids] = motion_ids
        self.motion_times[rewind_env_ids] = rewind_times

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
            self.motion_times[env_ids] = self.motion.sample_start_times(self.motion_ids[env_ids])
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        elif self.cfg.sampling_mode == "rewind":
            self._rewind_sampling(env_ids)
        else:
            raise ValueError(
                f"Unsupported motion sampling_mode={self.cfg.sampling_mode!r}. "
                "Supported modes are 'uniform', 'start', and 'rewind'."
            )

        self._reset_envs_to_current_reference(env_ids)

    def _reset_envs_to_current_reference(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        self._refresh_frame_cache()

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [
            self.cfg.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [
            self.cfg.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(
            lower=self.cfg.joint_position_range[0],
            upper=self.cfg.joint_position_range[1],
            size=joint_pos.shape,
            device=joint_pos.device,  # type: ignore
        )
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )

        root_state = torch.cat(
            [
                root_pos[env_ids],
                root_ori[env_ids],
                root_lin_vel[env_ids],
                root_ang_vel[env_ids],
            ],
            dim=-1,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        self.robot.clear_state(env_ids=env_ids)

        self._refresh_body_local_cache()
        self._update_feet_standing()

    def _refresh_body_local_cache(self) -> None:
        """Recompute body targets in the reference anchor's yaw-only local frame."""
        ref_yaw_inv = quat_inv(yaw_quat(self.anchor_quat_w))
        ref_yaw_inv_rep = ref_yaw_inv[:, None, :].expand_as(self.body_quat_w)
        anchor_pos_w_for_local = self.anchor_pos_w[:, None, :].expand_as(self.body_pos_w)

        self.body_pos_b = quat_apply(
            ref_yaw_inv_rep, self.body_pos_w - anchor_pos_w_for_local
        )
        self.body_quat_b = quat_mul(ref_yaw_inv_rep, self.body_quat_w)
        self.body_lin_vel_b = quat_apply(ref_yaw_inv_rep, self.body_lin_vel_w)
        self.body_ang_vel_b = quat_apply(ref_yaw_inv_rep, self.body_ang_vel_w)

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def _update_command(self):
        # Advance motion time by real elapsed time
        self.motion_times += self._step_dt
        if self.cfg.cache_swap_interval_steps > 0:
            self._motion_cache_step_counter += 1
            if self._motion_cache_step_counter >= self.cfg.cache_swap_interval_steps:
                self._motion_cache_swap_pending = True

        # Handle clips that exceeded their duration
        end_times = self.motion.clip_sample_end_s[self.motion_ids]
        exceeded = self.motion_times >= end_times

        env_ids = torch.where(exceeded)[0]
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

        self._refresh_frame_cache()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        self._refresh_body_local_cache()
        self._update_feet_standing()

    def apply_cache_swap_if_pending_barrier(self) -> bool:
        """Swap the staged motion cache at a rollout barrier, then resample all envs."""
        if not self._motion_cache_swap_pending:
            return False
        self.motion.advance_cache()
        self._motion_cache_step_counter = 0
        self._motion_cache_swap_pending = False
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if self.cfg.sampling_mode == "start":
            self.motion_ids[all_env_ids] = self.motion.sample_motion_ids(self.num_envs)
            self.motion_times[all_env_ids] = self.motion.sample_start_times(self.motion_ids[all_env_ids])
        else:
            # Rewind only makes sense inside one cache generation.  After a cache
            # swap, local ids refer to different clips, so fall back to uniform.
            self._uniform_sampling(all_env_ids)
        self._reset_envs_to_current_reference(all_env_ids)
        return True

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        """Draw ghost robot or frames based on visualization mode."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        if self.cfg.viz.mode == "ghost":
            if self._ghost_model is None:
                self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
                self._ghost_model.geom_rgba[:] = self._ghost_color

            entity: Entity = self._env.scene[self.cfg.entity_name]
            indexing = entity.indexing
            free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
            joint_q_adr = indexing.joint_q_adr.cpu().numpy()

            for batch in env_indices:
                qpos = np.zeros(self._env.sim.mj_model.nq)
                qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
                qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
                qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

                visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

        elif self.cfg.viz.mode == "frames":
            for batch in env_indices:
                desired_body_pos = self.body_pos_w[batch].cpu().numpy()
                desired_body_quat = self.body_quat_w[batch]
                desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

                current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
                current_body_quat = self.robot_body_quat_w[batch]
                current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

                for i, body_name in enumerate(self.cfg.body_names):
                    visualizer.add_frame(
                        position=desired_body_pos[i],
                        rotation_matrix=desired_body_rotm[i],
                        scale=0.08,
                        label=f"desired_{body_name}_{batch}",
                        axis_colors=_DESIRED_FRAME_COLORS,
                    )
                    visualizer.add_frame(
                        position=current_body_pos[i],
                        rotation_matrix=current_body_rotm[i],
                        scale=0.12,
                        label=f"current_{body_name}_{batch}",
                    )

                desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
                desired_anchor_quat = self.anchor_quat_w[batch]
                desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=desired_anchor_pos,
                    rotation_matrix=desired_rotation_matrix,
                    scale=0.1,
                    label=f"desired_anchor_{batch}",
                    axis_colors=_DESIRED_FRAME_COLORS,
                )

                current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
                current_anchor_quat = self.robot_anchor_quat_w[batch]
                current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
                visualizer.add_frame(
                    position=current_anchor_pos,
                    rotation_matrix=current_rotation_matrix,
                    scale=0.15,
                    label=f"current_anchor_{batch}",
                )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    motion_file: str
    anchor_body_name: str
    body_names: tuple[str, ...]
    entity_name: str
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    sampling_mode: Literal["uniform", "start", "rewind"] = "rewind"
    window_steps: tuple[int, ...] = (0,)
    cache_num_clips: int = 8192
    cache_swap_interval_steps: int = 2000
    cache_dataloader_num_workers: int = 2
    cache_dataloader_prefetch_factor: int = 1
    cache_dataloader_pin_memory: bool = True
    cache_seed: int = 0
    rewind_prob: float = 0.8
    rewind_min_steps: int = 25
    rewind_max_steps: int = 75
    feet_body_names: tuple[str, ...] = ()
    feet_standing_z_threshold: float = 0.18
    feet_standing_vxy_threshold: float = 0.2
    feet_standing_vz_threshold: float = 0.15

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
