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
            clip_sample_starts, clip_sample_ends = compute_clip_sample_ranges(
                self.clip_lengths.cpu().numpy(),
                window_steps=self.window_steps,
            )
            self.clip_sample_starts = torch.tensor(
                clip_sample_starts, dtype=torch.long, device=device
            )
            self.clip_sample_ends = torch.tensor(
                clip_sample_ends, dtype=torch.long, device=device
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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

        idx0 = starts[:, None] + frame_i0
        idx1 = starts[:, None] + frame_i1
        window = len(steps)
        return (
            idx0.cpu().numpy().reshape(-1),
            idx1.cpu().numpy().reshape(-1),
            alpha.cpu().numpy().reshape(-1),
            window,
        )

    def get_window_frames(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        *,
        window_steps: tuple[int, ...] | list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = parse_window_steps(self.window_steps if window_steps is None else window_steps)
        idx0_np, idx1_np, alpha_np, window = self._compute_interpolation_state(
            motion_ids,
            motion_times,
            steps,
        )
        batch = motion_ids.shape[0]

        result: dict[str, torch.Tensor] = {}
        a1 = alpha_np[:, None]
        for key, arr in (("joint_pos", self._joint_pos), ("joint_vel", self._joint_vel)):
            v0, v1 = arr[idx0_np], arr[idx1_np]
            interp = v0 + a1 * (v1 - v0)
            result[key] = torch.from_numpy(interp.reshape(batch, window, -1)).to(
                self._device,
                non_blocking=True,
            )

        a2 = alpha_np[:, None, None]
        for key, arr in (
            ("body_pos_w", self._body_pos_w),
            ("body_lin_vel_w", self._body_lin_vel_w),
            ("body_ang_vel_w", self._body_ang_vel_w),
        ):
            v0, v1 = arr[idx0_np], arr[idx1_np]
            interp = v0 + a2 * (v1 - v0)
            result[key] = torch.from_numpy(interp.reshape(batch, window, *interp.shape[1:])).to(
                self._device,
                non_blocking=True,
            )

        q0 = torch.from_numpy(np.ascontiguousarray(self._body_quat_w[idx0_np]))
        q1 = torch.from_numpy(np.ascontiguousarray(self._body_quat_w[idx1_np]))
        alpha_t = torch.from_numpy(alpha_np)
        nb = q0.shape[1]
        q0_flat = q0.reshape(batch * window * nb, 4)
        q1_flat = q1.reshape(batch * window * nb, 4)
        alpha_flat = alpha_t.unsqueeze(-1).expand(batch * window, nb).reshape(batch * window * nb)
        result["body_quat_w"] = (
            _batched_quat_slerp(q0_flat, q1_flat, alpha_flat)
            .reshape(batch, window, nb, 4)
            .to(self._device, non_blocking=True)
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
        Frame indices are computed on GPU (where motion_ids / motion_times
        live), then transferred to CPU for numpy-based indexing and
        interpolation.  The small interpolated result is sent back to GPU.
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
        else:
            self._uniform_sampling(env_ids)

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
    sampling_mode: Literal["uniform", "start"] = "uniform"
    window_steps: tuple[int, ...] = (0,)

    @dataclass
    class VizCfg:
        mode: Literal["ghost", "frames"] = "ghost"
        ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
        return MotionCommand(self, env)
