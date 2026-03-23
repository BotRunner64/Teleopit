from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from train_mimic.data.dataset_lib import compute_clip_sample_ranges, parse_window_steps

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


def _compute_clip_counts(
    motion_ids: torch.Tensor, episode_failed: torch.Tensor, bin_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (exposure_count, failure_count) per clip bin."""
    exposure = torch.bincount(motion_ids, minlength=bin_count).float()
    failure = torch.bincount(
        motion_ids[episode_failed], minlength=bin_count
    ).float()
    return exposure, failure


def _compute_clip_failure_rate(
    motion_ids: torch.Tensor, episode_failed: torch.Tensor, bin_count: int
) -> torch.Tensor:
    """Compute per-clip failure rate for the current adaptive-sampling window."""
    exposure, failure = _compute_clip_counts(motion_ids, episode_failed, bin_count)
    rate = torch.zeros(bin_count, dtype=torch.float32, device=motion_ids.device)
    valid = exposure > 0
    rate[valid] = failure[valid] / exposure[valid]
    return rate


def _is_distributed() -> bool:
    """Return True when running inside an initialized torch.distributed group."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _normalize_sampling_probabilities(
    sampling_probabilities: torch.Tensor,
    *,
    adaptive_uniform_ratio: float,
    bin_count: int,
) -> torch.Tensor:
    """Normalize adaptive sampling weights and fail fast on invalid mass."""
    prob_sum = sampling_probabilities.sum()
    if not torch.isfinite(prob_sum) or prob_sum <= 0:
        raise ValueError(
            "Adaptive sampling produced an invalid probability mass. "
            f"sum={prob_sum.item() if torch.isfinite(prob_sum) else prob_sum}, "
            f"adaptive_uniform_ratio={adaptive_uniform_ratio}, bin_count={bin_count}. "
            "Increase adaptive_uniform_ratio or accumulate failure statistics before "
            "using pure adaptive sampling."
        )

    sampling_probabilities = sampling_probabilities / prob_sum
    if not torch.isfinite(sampling_probabilities).all():
        raise ValueError("Adaptive sampling probabilities contain NaN or Inf values.")
    return sampling_probabilities


def _validate_legacy_adaptive_config(*, adaptive_kernel_size: int, adaptive_lambda: float) -> None:
    """Reject legacy adaptive knobs that are no longer implemented."""
    if adaptive_kernel_size != 1:
        raise ValueError(
            "adaptive_kernel_size is not implemented in the restored adaptive sampler. "
            f"Expected 1, got {adaptive_kernel_size}."
        )
    if adaptive_lambda != 0.8:
        raise ValueError(
            "adaptive_lambda is not implemented in the restored adaptive sampler. "
            f"Expected 0.8, got {adaptive_lambda}."
        )


class MotionLib:
    """Clip-aware motion library.

    Loads a merged NPZ that contains flat motion arrays plus per-clip metadata
    (``clip_starts``, ``clip_lengths``, ``clip_fps``, ``clip_weights``).  Falls
    back to single-clip mode when the metadata keys are absent (legacy format).

    Motion data is stored as GPU tensors for fast gather+lerp interpolation.
    All indexing, lerp, and slerp run entirely on device with zero CPU
    round-trips.  Numpy arrays are kept alongside for external consumers
    (e.g. runner checkpoint buffers).
    """

    def __init__(
        self,
        motion_file: str,
        body_indexes: torch.Tensor,
        device: str = "cpu",
        window_steps: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        self._device = device
        self.window_steps = parse_window_steps(window_steps)

        # .npz is a zip archive — mmap_mode is ignored by np.load for .npz.
        # We load one array at a time and immediately body-filter + keep only
        # the numpy result, so the full unfiltered array can be GC'd before
        # the next one is loaded.
        data = np.load(motion_file, allow_pickle=True)
        body_idx_np = body_indexes.cpu().numpy()

        self._joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)  # (T, 29)
        self._joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)  # (T, 29)

        # Body arrays: index by selected bodies immediately.  Accessing an
        # NpzFile key inflates that array from the zip; the intermediate full
        # array is released once we slice and discard the reference.
        self._body_pos_w = np.asarray(
            data["body_pos_w"], dtype=np.float32
        )[:, body_idx_np]
        self._body_quat_w = np.asarray(
            data["body_quat_w"], dtype=np.float32
        )[:, body_idx_np]
        self._body_lin_vel_w = np.asarray(
            data["body_lin_vel_w"], dtype=np.float32
        )[:, body_idx_np]
        self._body_ang_vel_w = np.asarray(
            data["body_ang_vel_w"], dtype=np.float32
        )[:, body_idx_np]

        self.time_step_total = self._joint_pos.shape[0]

        # GPU tensors for fast gather+lerp interpolation (zero CPU round-trips).
        self._joint_pos_t = torch.from_numpy(self._joint_pos).to(device)
        self._joint_vel_t = torch.from_numpy(self._joint_vel).to(device)
        self._body_pos_w_t = torch.from_numpy(self._body_pos_w).to(device)
        self._body_quat_w_t = torch.from_numpy(self._body_quat_w).to(device)
        self._body_lin_vel_w_t = torch.from_numpy(self._body_lin_vel_w).to(device)
        self._body_ang_vel_w_t = torch.from_numpy(self._body_ang_vel_w).to(device)

        # --- clip-aware metadata (small — lives on GPU for sampling) ---
        if "clip_starts" in data:
            self.clip_starts = torch.tensor(data["clip_starts"], dtype=torch.long, device=device)
            self.clip_lengths = torch.tensor(data["clip_lengths"], dtype=torch.long, device=device)
            self.clip_weights = torch.tensor(
                data["clip_weights"], dtype=torch.float32, device=device
            )
            fps_arr = np.asarray(data["clip_fps"])
            if fps_arr.ndim == 0:
                self.clip_fps = torch.full(
                    (len(self.clip_starts),), float(fps_arr), dtype=torch.float32, device=device
                )
            else:
                self.clip_fps = torch.tensor(fps_arr, dtype=torch.float32, device=device)
        else:
            # Legacy single-clip fallback
            fps_scalar = float(data["fps"])
            self.clip_starts = torch.tensor([0], dtype=torch.long, device=device)
            self.clip_lengths = torch.tensor(
                [self.time_step_total], dtype=torch.long, device=device
            )
            self.clip_weights = torch.tensor([1.0], dtype=torch.float32, device=device)
            self.clip_fps = torch.tensor([fps_scalar], dtype=torch.float32, device=device)

        self.num_clips = len(self.clip_starts)
        self.clip_dt = 1.0 / self.clip_fps
        self.clip_duration_s = (self.clip_lengths.float() - 1.0) * self.clip_dt

        # Compute minimum clip length required by the window
        max_future = max((s for s in self.window_steps if s > 0), default=0)
        max_history = -min((s for s in self.window_steps if s < 0), default=0)
        min_clip_length = max_history + 1 + max_future + 1  # +1 for interpolation
        short_mask = self.clip_lengths < min_clip_length
        n_short = int(short_mask.sum().item())
        if n_short > 0:
            _LOG.warning(
                "Disabling %d/%d clips shorter than %d frames (window_steps=%s)",
                n_short, self.num_clips, min_clip_length, list(self.window_steps),
            )
            self.clip_weights = self.clip_weights.clone()
            self.clip_weights[short_mask] = 0.0

        file_window_steps = parse_window_steps(data["window_steps"]) if "window_steps" in data else (0,)
        if (
            "clip_sample_starts" in data
            and "clip_sample_ends" in data
            and file_window_steps == self.window_steps
        ):
            self.clip_sample_starts = torch.tensor(
                data["clip_sample_starts"], dtype=torch.long, device=device
            )
            self.clip_sample_ends = torch.tensor(
                data["clip_sample_ends"], dtype=torch.long, device=device
            )
        else:
            # Only compute ranges for clips long enough; short ones get dummy [0,1)
            lengths_np = self.clip_lengths.cpu().numpy()
            sample_starts = np.zeros(self.num_clips, dtype=np.int64)
            sample_ends = np.ones(self.num_clips, dtype=np.int64)
            long_mask = ~short_mask.cpu().numpy()
            if long_mask.any():
                s, e = compute_clip_sample_ranges(
                    lengths_np[long_mask],
                    window_steps=self.window_steps,
                )
                sample_starts[long_mask] = s
                sample_ends[long_mask] = e
            self.clip_sample_starts = torch.tensor(
                sample_starts, dtype=torch.long, device=device
            )
            self.clip_sample_ends = torch.tensor(
                sample_ends, dtype=torch.long, device=device
            )
        self.clip_sample_start_s = self.clip_sample_starts.float() * self.clip_dt
        self.clip_sample_end_s = self.clip_sample_ends.float() * self.clip_dt

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        """Sample *n* clip indices weighted by ``clip_weights``."""
        total = self.clip_weights.sum()
        if total <= 0:
            raise ValueError(
                "All clip weights are zero — cannot sample. "
                "Check that the merged NPZ was built with positive weights."
            )
        probs = self.clip_weights / total
        return torch.multinomial(probs, n, replacement=True)

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
        starts = self.clip_starts[motion_ids]
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

        idx0 = (starts[:, None] + frame_i0).reshape(-1)
        idx1 = (starts[:, None] + frame_i1).reshape(-1)
        window = len(steps)
        return idx0, idx1, alpha.reshape(-1), window

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
        want = self._ALL_KEYS if keys is None else keys

        result: dict[str, torch.Tensor] = {}

        # joint_pos, joint_vel: (T, D) → GPU gather + lerp
        a1 = alpha[:, None]
        for key, arr_t in (("joint_pos", self._joint_pos_t), ("joint_vel", self._joint_vel_t)):
            if key not in want:
                continue
            v0, v1 = arr_t[idx0], arr_t[idx1]
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
                v0, v1 = arr_t[idx0][:, body_indices], arr_t[idx1][:, body_indices]
            else:
                v0, v1 = arr_t[idx0], arr_t[idx1]
            interp = v0 + a2 * (v1 - v0)
            result[key] = interp.reshape(batch, window, *interp.shape[1:])

        # body_quat_w: GPU slerp, optionally pre-slice bodies
        if "body_quat_w" in want:
            if body_indices is not None:
                q0 = self._body_quat_w_t[idx0][:, body_indices]
                q1 = self._body_quat_w_t[idx1][:, body_indices]
            else:
                q0 = self._body_quat_w_t[idx0]
                q1 = self._body_quat_w_t[idx1]
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


# Backward compatibility alias
MotionLoader = MotionLib


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        _validate_legacy_adaptive_config(
            adaptive_kernel_size=cfg.adaptive_kernel_size,
            adaptive_lambda=cfg.adaptive_lambda,
        )

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
            device=self.device,
            window_steps=self.cfg.window_steps,
        )

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

        self.bin_count = max(self.motion.num_clips, 1)
        self.bin_failed_rate = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._accum_exposure_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._accum_failure_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self.kernel = torch.ones(1, device=self.device)

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
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        # Feet standing state (for feet_air_time_ref rewards)
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

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        current_motion_ids = self.motion_ids[env_ids]
        episode_failed = self._env.termination_manager.terminated[env_ids]
        exposure, failure = _compute_clip_counts(
            current_motion_ids, episode_failed, self.bin_count
        )
        self._accum_exposure_count += exposure
        self._accum_failure_count += failure

        sampling_probabilities = (
            self.bin_failed_rate
            + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )
        sampling_probabilities = _normalize_sampling_probabilities(
            sampling_probabilities,
            adaptive_uniform_ratio=self.cfg.adaptive_uniform_ratio,
            bin_count=self.bin_count,
        )

        sampled_clips = torch.multinomial(
            sampling_probabilities, len(env_ids), replacement=True
        )
        self.motion_ids[env_ids] = sampled_clips
        self.motion_times[env_ids] = self.motion.sample_times(sampled_clips)

        entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        entropy_norm = entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        top1_prob, top1_bin = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = entropy_norm
        self.metrics["sampling_top1_prob"][:] = top1_prob
        self.metrics["sampling_top1_bin"][:] = top1_bin.float() / self.bin_count

    def _uniform_sampling(self, env_ids: torch.Tensor):
        self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
        self.motion_times[env_ids] = self.motion.sample_times(self.motion_ids[env_ids])
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.bin_count, 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
            self.motion_times[env_ids] = self.motion.sample_start_times(self.motion_ids[env_ids])
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            assert self.cfg.sampling_mode == "adaptive"
            self._adaptive_sampling(env_ids)

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

        if self.cfg.sampling_mode == "adaptive":
            # Sync raw counts across ranks for unified statistics
            if _is_distributed():
                torch.distributed.all_reduce(self._accum_exposure_count)
                torch.distributed.all_reduce(self._accum_failure_count)

            # Only update EMA when new data exists (fixes decay-on-empty-step)
            if self._accum_exposure_count.sum() > 0:
                valid = self._accum_exposure_count > 0
                global_rate = torch.zeros_like(self.bin_failed_rate)
                global_rate[valid] = (
                    self._accum_failure_count[valid]
                    / self._accum_exposure_count[valid]
                )
                self.bin_failed_rate = (
                    self.cfg.adaptive_alpha * global_rate
                    + (1 - self.cfg.adaptive_alpha) * self.bin_failed_rate
                )

            self._accum_exposure_count.zero_()
            self._accum_failure_count.zero_()

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
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    sampling_mode: Literal["adaptive", "uniform", "start"] = "uniform"
    window_steps: tuple[int, ...] = (0,)
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
