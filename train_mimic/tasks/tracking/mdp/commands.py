from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

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


def _compute_clip_failure_counts(
    motion_ids: torch.Tensor, episode_failed: torch.Tensor, bin_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-clip exposure and failure counts for one adaptive window."""
    exposure_count = torch.bincount(motion_ids, minlength=bin_count).float()
    failure_count = torch.bincount(
        motion_ids[episode_failed], minlength=bin_count
    ).float()
    return exposure_count, failure_count


def _compute_failure_rate_from_counts(
    exposure_count: torch.Tensor, failure_count: torch.Tensor
) -> torch.Tensor:
    """Convert per-clip exposure and failure counts into failure rates."""
    failure_rate = torch.zeros_like(exposure_count, dtype=torch.float32)
    valid = exposure_count > 0
    failure_rate[valid] = failure_count[valid] / exposure_count[valid]
    return failure_rate


def _compute_clip_failure_rate(
    motion_ids: torch.Tensor, episode_failed: torch.Tensor, bin_count: int
) -> torch.Tensor:
    """Compute per-clip failure rate for the current adaptive-sampling window."""
    exposure_count, failure_count = _compute_clip_failure_counts(
        motion_ids, episode_failed, bin_count
    )
    return _compute_failure_rate_from_counts(exposure_count, failure_count)


def _compute_rank_sample_range(request_counts: torch.Tensor, rank: int) -> tuple[int, int]:
    """Return the [start, end) slice in a global sample tensor for one rank."""
    if request_counts.dim() != 1:
        raise ValueError(f"request_counts must be 1-D, got shape {tuple(request_counts.shape)}")
    if rank < 0 or rank >= int(request_counts.numel()):
        raise ValueError(
            f"rank {rank} is out of range for request_counts with {request_counts.numel()} entries"
        )
    if request_counts.numel() == 0:
        return 0, 0
    prefix = torch.cumsum(request_counts, dim=0)
    start = int(prefix[rank - 1].item()) if rank > 0 else 0
    end = int(prefix[rank].item())
    return start, end


def _compute_windowed_ema_alpha(alpha: float, num_steps: int) -> float:
    """Convert a per-step EMA alpha into an equivalent ``num_steps`` window alpha."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return 1.0 - math.pow(1.0 - alpha, num_steps)


class MotionLib:
    """Clip-aware motion library.

    Loads a merged NPZ that contains flat motion arrays plus per-clip metadata
    (``clip_starts``, ``clip_lengths``, ``clip_fps``, ``clip_weights``).  Falls
    back to single-clip mode when the metadata keys are absent (legacy format).

    Motion data is kept on CPU (numpy arrays) to avoid GPU OOM on large
    datasets.  Only the small per-batch interpolated frames are transferred to
    GPU each step (~6 MB for 4096 envs, negligible vs PCIe bandwidth).
    """

    def __init__(
        self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
    ) -> None:
        self._device = device

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
        """Uniform random time in [0, clip_duration_s) for each motion id."""
        durations = self.clip_duration_s[motion_ids]
        return torch.rand_like(durations) * durations

    # ------------------------------------------------------------------
    # Frame interpolation
    # ------------------------------------------------------------------

    def get_frames(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Look up interpolated motion state for ``(motion_id, time)`` pairs.

        Handles time wrapping (loop) and sub-frame linear / slerp interpolation.
        Frame indices are computed on GPU (where motion_ids / motion_times
        live), then transferred to CPU for numpy-based indexing and
        interpolation.  The small interpolated result is sent back to GPU.
        """
        fps = self.clip_fps[motion_ids]
        starts = self.clip_starts[motion_ids]
        lengths = self.clip_lengths[motion_ids]
        durations = self.clip_duration_s[motion_ids]

        # Wrap for looping
        safe_dur = torch.clamp(durations, min=1e-6)
        motion_times = torch.fmod(motion_times, safe_dur)
        motion_times = torch.where(motion_times < 0, motion_times + safe_dur, motion_times)

        # Fractional frame index within clip
        frame_f = motion_times * fps
        frame_i0 = frame_f.long()
        frame_i1 = frame_i0 + 1
        alpha = frame_f - frame_i0.float()

        max_frame = lengths - 1
        frame_i0 = torch.clamp(frame_i0, min=torch.zeros_like(frame_i0), max=max_frame)
        frame_i1 = torch.clamp(frame_i1, min=torch.zeros_like(frame_i1), max=max_frame)

        idx0 = starts + frame_i0
        idx1 = starts + frame_i1

        # --- Transfer indices to CPU for numpy indexing ---
        idx0_np = idx0.cpu().numpy()
        idx1_np = idx1.cpu().numpy()
        alpha_np = alpha.cpu().numpy()

        result: dict[str, torch.Tensor] = {}

        # Linear interpolation for 2-D arrays (n, D)
        a1 = alpha_np[:, None]
        for key, arr in (("joint_pos", self._joint_pos), ("joint_vel", self._joint_vel)):
            v0, v1 = arr[idx0_np], arr[idx1_np]
            result[key] = torch.from_numpy(v0 + a1 * (v1 - v0)).to(
                self._device, non_blocking=True
            )

        # Linear interpolation for 3-D arrays (n, B, D)
        a2 = alpha_np[:, None, None]
        for key, arr in (
            ("body_pos_w", self._body_pos_w),
            ("body_lin_vel_w", self._body_lin_vel_w),
            ("body_ang_vel_w", self._body_ang_vel_w),
        ):
            v0, v1 = arr[idx0_np], arr[idx1_np]
            result[key] = torch.from_numpy(v0 + a2 * (v1 - v0)).to(
                self._device, non_blocking=True
            )

        # Slerp for quaternions (n, B, 4) — done in torch on CPU, then to GPU
        q0 = torch.from_numpy(np.ascontiguousarray(self._body_quat_w[idx0_np]))
        q1 = torch.from_numpy(np.ascontiguousarray(self._body_quat_w[idx1_np]))
        alpha_t = torch.from_numpy(alpha_np)
        n, nb = q0.shape[0], q0.shape[1]
        q0_flat = q0.reshape(n * nb, 4)
        q1_flat = q1.reshape(n * nb, 4)
        alpha_flat = alpha_t.unsqueeze(-1).expand(n, nb).reshape(n * nb)
        result["body_quat_w"] = (
            _batched_quat_slerp(q0_flat, q1_flat, alpha_flat)
            .reshape(n, nb, 4)
            .to(self._device, non_blocking=True)
        )

        return result


# Backward compatibility alias
MotionLoader = MotionLib


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg
    _env: ManagerBasedRlEnv

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

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
            self.cfg.motion_file, self.body_indexes, device=self.device
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

        # Adaptive sampling — bins are now per-clip and track an EMA of
        # failure rate rather than raw failure count to avoid scaling the
        # adaptive signal with num_envs.
        self.bin_count = max(self.motion.num_clips, 1)
        self.bin_failed_rate = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._adaptive_pending_exposure_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._adaptive_pending_failure_count = torch.zeros(
            self.bin_count, dtype=torch.float, device=self.device
        )
        self._adaptive_step_counter = 0
        self._adaptive_sync_counter = 0
        self._adaptive_resample_counter = 0
        self._adaptive_sync_alpha = _compute_windowed_ema_alpha(
            self.cfg.adaptive_alpha, self.cfg.adaptive_sync_steps
        )
        self.kernel = torch.ones(1, device=self.device)

        if self.cfg.adaptive_sync_steps <= 0:
            raise ValueError(
                f"adaptive_sync_steps must be positive, got {self.cfg.adaptive_sync_steps}"
            )
        if self.cfg.adaptive_log_interval <= 0:
            raise ValueError(
                f"adaptive_log_interval must be positive, got {self.cfg.adaptive_log_interval}"
            )

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

        # Ghost model created lazily on first visualization
        self._ghost_model: mujoco.MjModel | None = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # ------------------------------------------------------------------
    # Frame cache
    # ------------------------------------------------------------------

    def _refresh_frame_cache(self) -> None:
        self._cached_frames = self.motion.get_frames(self.motion_ids, self.motion_times)

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

    def _sampling_probabilities(self) -> torch.Tensor:
        sampling_probabilities = (
            self.bin_failed_rate + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        )
        total = sampling_probabilities.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError(
                "Adaptive sampling produced an invalid probability mass. "
                f"sum={total.item() if torch.isfinite(total) else total}, "
                f"uniform_ratio={self.cfg.adaptive_uniform_ratio}, bin_count={self.bin_count}"
            )
        sampling_probabilities = sampling_probabilities / total
        if not torch.isfinite(sampling_probabilities).all():
            raise ValueError("Adaptive sampling probabilities contain NaN or Inf values.")
        return sampling_probabilities

    def _update_sampling_metrics(self, sampling_probabilities: torch.Tensor) -> None:
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _distributed_runtime(self) -> tuple[object | None, bool, int, int]:
        distributed = getattr(torch, "distributed", None)
        enabled = bool(
            distributed is not None
            and distributed.is_available()
            and distributed.is_initialized()
        )
        if not enabled or distributed is None:
            return distributed, False, 0, 1
        return distributed, True, int(distributed.get_rank()), int(distributed.get_world_size())

    def _gather_adaptive_request_counts(self, local_count: int) -> torch.Tensor:
        request_count = torch.tensor([local_count], dtype=torch.long, device=self.device)
        distributed, dist_enabled, _dist_rank, dist_world_size = self._distributed_runtime()
        if not dist_enabled or distributed is None:
            return request_count

        gathered = [torch.zeros_like(request_count) for _ in range(dist_world_size)]
        distributed.all_gather(gathered, request_count)
        return torch.cat(gathered)

    def _maybe_log_adaptive_batch(
        self,
        request_counts: torch.Tensor,
        sampling_probabilities: torch.Tensor,
        sampled_clips: torch.Tensor,
    ) -> None:
        _distributed, _dist_enabled, dist_rank, _dist_world_size = self._distributed_runtime()
        if dist_rank != 0:
            return

        should_log = self._adaptive_resample_counter % self.cfg.adaptive_log_interval == 0
        if not should_log:
            top1_prob = float(sampling_probabilities.max().item())
            entropy = float(self.metrics["sampling_entropy"][0].item())
            should_log = (
                top1_prob >= self.cfg.adaptive_warn_top1_prob
                or entropy <= self.cfg.adaptive_warn_entropy
            )
        if not should_log:
            return

        sample_hist = torch.bincount(sampled_clips, minlength=self.bin_count).float()
        topk = min(self.cfg.adaptive_log_topk, self.bin_count)
        top_counts, top_indices = torch.topk(sample_hist, k=topk)
        parts = []
        total_samples = max(int(sampled_clips.numel()), 1)
        for clip_id, count in zip(top_indices.tolist(), top_counts.tolist(), strict=False):
            if count <= 0:
                continue
            parts.append(
                f"{clip_id}:count={int(count)},share={count / total_samples:.4f},"
                f"prob={float(sampling_probabilities[clip_id].item()):.4f}"
            )

        _LOG.warning(
            "[adaptive_sampling_batch] batch=%d env_step=%d total_resamples=%d "
            "request_counts=%s top_draws=[%s]",
            self._adaptive_resample_counter,
            self._adaptive_step_counter,
            int(sampled_clips.numel()),
            request_counts.tolist(),
            ", ".join(parts),
        )

    def _maybe_log_adaptive_sync(
        self,
        observed_failure_rate: torch.Tensor,
        exposure_count: torch.Tensor,
        sampling_probabilities: torch.Tensor,
    ) -> None:
        _distributed, _dist_enabled, dist_rank, dist_world_size = self._distributed_runtime()
        if dist_rank != 0:
            return

        should_log = self._adaptive_sync_counter % self.cfg.adaptive_log_interval == 0
        if not should_log:
            top1_prob = float(sampling_probabilities.max().item())
            entropy = float(self.metrics["sampling_entropy"][0].item())
            should_log = (
                top1_prob >= self.cfg.adaptive_warn_top1_prob
                or entropy <= self.cfg.adaptive_warn_entropy
            )
        if not should_log:
            return

        topk = min(self.cfg.adaptive_log_topk, self.bin_count)
        top_probs, top_indices = torch.topk(sampling_probabilities, k=topk)
        parts = []
        for clip_id, prob in zip(top_indices.tolist(), top_probs.tolist(), strict=False):
            fail_rate = float(observed_failure_rate[clip_id].item())
            exposures = int(exposure_count[clip_id].item())
            parts.append(
                f"{clip_id}:p={prob:.4f},fail={fail_rate:.4f},exp={exposures}"
            )

        _LOG.warning(
            "[adaptive_sampling] sync=%d env_step=%d world_size=%d active_clips=%d "
            "entropy=%.4f top1_prob=%.4f top_bins=[%s]",
            self._adaptive_sync_counter,
            self._adaptive_step_counter,
            dist_world_size,
            int((exposure_count > 0).sum().item()),
            float(self.metrics["sampling_entropy"][0].item()),
            float(self.metrics["sampling_top1_prob"][0].item()),
            ", ".join(parts),
        )

    def _sync_adaptive_failure_rate(self) -> None:
        exposure_count = self._adaptive_pending_exposure_count.clone()
        failure_count = self._adaptive_pending_failure_count.clone()
        distributed, dist_enabled, _dist_rank, _dist_world_size = self._distributed_runtime()
        if dist_enabled and distributed is not None:
            distributed.all_reduce(exposure_count, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(failure_count, op=distributed.ReduceOp.SUM)

        observed_failure_rate = _compute_failure_rate_from_counts(
            exposure_count, failure_count
        )
        self.bin_failed_rate = (
            self._adaptive_sync_alpha * observed_failure_rate
            + (1 - self._adaptive_sync_alpha) * self.bin_failed_rate
        )
        self._adaptive_pending_exposure_count.zero_()
        self._adaptive_pending_failure_count.zero_()
        self._adaptive_sync_counter += 1

        sampling_probabilities = self._sampling_probabilities()
        self._update_sampling_metrics(sampling_probabilities)
        self._maybe_log_adaptive_sync(
            observed_failure_rate=observed_failure_rate,
            exposure_count=exposure_count,
            sampling_probabilities=sampling_probabilities,
        )

    def _adaptive_resample_motion_ids(self, env_ids: torch.Tensor) -> None:
        current_motion_ids = self.motion_ids[env_ids]
        episode_failed = self._env.termination_manager.terminated[env_ids]
        exposure_count, failure_count = _compute_clip_failure_counts(
            current_motion_ids, episode_failed, self.bin_count
        )
        self._adaptive_pending_exposure_count += exposure_count
        self._adaptive_pending_failure_count += failure_count

        request_counts = self._gather_adaptive_request_counts(int(env_ids.numel()))
        total_samples = int(request_counts.sum().item())
        if total_samples == 0:
            return

        sampling_probabilities = self._sampling_probabilities()
        self._update_sampling_metrics(sampling_probabilities)

        sampled_clips = torch.empty(total_samples, dtype=torch.long, device=self.device)
        distributed, dist_enabled, dist_rank, _dist_world_size = self._distributed_runtime()
        if dist_rank == 0:
            sampled_clips.copy_(
                torch.multinomial(
                    sampling_probabilities, total_samples, replacement=True
                )
            )
            self._adaptive_resample_counter += 1
            self._maybe_log_adaptive_batch(
                request_counts=request_counts,
                sampling_probabilities=sampling_probabilities,
                sampled_clips=sampled_clips,
            )
        if dist_enabled and distributed is not None:
            distributed.broadcast(sampled_clips, src=0)
            batch_counter = torch.tensor(
                [self._adaptive_resample_counter], dtype=torch.long, device=self.device
            )
            distributed.broadcast(batch_counter, src=0)
            self._adaptive_resample_counter = int(batch_counter.item())

        start, end = _compute_rank_sample_range(request_counts, dist_rank)
        local_sampled_clips = sampled_clips[start:end]
        if local_sampled_clips.numel() != env_ids.numel():
            raise RuntimeError(
                "Adaptive sample allocation mismatch across ranks: "
                f"rank={dist_rank}, allocated={local_sampled_clips.numel()}, "
                f"requested={env_ids.numel()}, request_counts={request_counts.tolist()}"
            )
        if env_ids.numel() > 0:
            self.motion_ids[env_ids] = local_sampled_clips
            self.motion_times[env_ids] = self.motion.sample_times(local_sampled_clips)

    def _uniform_sampling(self, env_ids: torch.Tensor):
        self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
        self.motion_times[env_ids] = self.motion.sample_times(self.motion_ids[env_ids])
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / max(self.bin_count, 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            self.motion_ids[env_ids] = self.motion.sample_motion_ids(len(env_ids))
            self.motion_times[env_ids] = 0.0
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            assert self.cfg.sampling_mode == "adaptive"
            self._adaptive_resample_motion_ids(env_ids)

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
        self._adaptive_step_counter += 1

        # Advance motion time by real elapsed time
        self.motion_times += self._step_dt

        # Handle clips that exceeded their duration
        durations = self.motion.clip_duration_s[self.motion_ids]
        exceeded = self.motion_times >= durations

        env_ids = torch.where(exceeded)[0]
        if self.cfg.sampling_mode == "adaptive":
            self._resample_command(env_ids)
        elif env_ids.numel() > 0:
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

        if (
            self.cfg.sampling_mode == "adaptive"
            and self._adaptive_step_counter % self.cfg.adaptive_sync_steps == 0
        ):
            self._sync_adaptive_failure_rate()

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
    adaptive_sync_steps: int = 24
    adaptive_log_interval: int = 50
    adaptive_log_topk: int = 5
    adaptive_warn_top1_prob: float = 0.2
    adaptive_warn_entropy: float = 0.6
    sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
