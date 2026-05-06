"""Shared reference processing algorithms for sim and sim2real paths.

Contains the core algorithms for retargeting conversion, yaw alignment,
anchor velocity computation, and observation dispatch that are used by
both PolicyStepRunner (sim) and Sim2RealReferenceProcessor (sim2real).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy.typing import NDArray

from teleopit.constants import FULL_QPOS_DIM
from teleopit.controllers.observation import (
    compute_fixed_yaw_alignment_quat,
    rotate_motion_qpos_by_yaw,
)
from teleopit.interfaces import SupportsReferenceWindow
from teleopit.math_utils import quat_inv_np, quat_mul_np
from teleopit.sim.reference_timeline import ReferenceSample, ReferenceWindow

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
logger = logging.getLogger(__name__)


def retarget_to_qpos(retargeted: object) -> Float64Array:
    """Convert retarget output to 36D qpos (7D root + 29D joints).

    Accepts either a (base_pos, base_rot, joint_pos) tuple or a flat array.
    """
    if isinstance(retargeted, tuple) and len(retargeted) == 3:
        base_pos = np.asarray(retargeted[0], dtype=np.float64).reshape(-1)
        base_rot = np.asarray(retargeted[1], dtype=np.float64).reshape(-1)
        joint_pos = np.asarray(retargeted[2], dtype=np.float64).reshape(-1)
        qpos = np.concatenate((base_pos, base_rot, joint_pos))
    else:
        qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
    if qpos.shape[0] < FULL_QPOS_DIM:
        raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= 36)")
    return qpos


def align_reference_yaw(
    qpos: Float64Array,
    robot_quat: Float32Array,
    fixed_yaw_quat: Float32Array | None,
    fixed_pivot_pos: Float32Array | None,
) -> tuple[Float64Array, Float32Array, Float32Array]:
    """Apply fixed yaw alignment to reference qpos.

    On first call (when fixed_yaw_quat is None), initializes the alignment
    from the robot's current orientation. Returns (aligned_qpos, yaw_quat, pivot_pos).
    """
    if fixed_yaw_quat is None:
        fixed_yaw_quat = compute_fixed_yaw_alignment_quat(
            robot_quat,
            np.asarray(qpos[3:7], dtype=np.float32),
        )
        fixed_pivot_pos = np.asarray(qpos[0:3], dtype=np.float32).copy()

    aligned_qpos = qpos.copy()
    rotate_motion_qpos_by_yaw(aligned_qpos, fixed_yaw_quat, fixed_pivot_pos)
    return aligned_qpos, fixed_yaw_quat, cast(Float32Array, fixed_pivot_pos)


def align_reference_window(
    reference_window: ReferenceWindow | None,
    fixed_ref_yaw_alignment: bool,
    robot_quat: Float32Array,
    fixed_yaw_quat: Float32Array | None,
    fixed_pivot_pos: Float32Array | None,
) -> tuple[ReferenceWindow | None, Float32Array | None, Float32Array | None]:
    """Align all samples in a reference window by fixed yaw.

    Returns (aligned_window, updated_yaw_quat, updated_pivot_pos).
    """
    if reference_window is None or not fixed_ref_yaw_alignment:
        return reference_window, fixed_yaw_quat, fixed_pivot_pos

    current_qpos = np.asarray(reference_window.current_sample().qpos, dtype=np.float64).reshape(-1)
    if fixed_yaw_quat is None:
        fixed_yaw_quat = compute_fixed_yaw_alignment_quat(
            robot_quat,
            np.asarray(current_qpos[3:7], dtype=np.float32),
        )
        fixed_pivot_pos = np.asarray(current_qpos[0:3], dtype=np.float32).copy()

    aligned_samples: list[ReferenceSample] = []
    for sample in reference_window.samples:
        aligned_qpos = np.asarray(sample.qpos, dtype=np.float64).reshape(-1).copy()
        rotate_motion_qpos_by_yaw(
            aligned_qpos,
            cast(Float32Array, fixed_yaw_quat),
            cast(Float32Array, fixed_pivot_pos),
        )
        aligned_samples.append(
            ReferenceSample(
                qpos=aligned_qpos,
                timestamp_s=float(sample.timestamp_s),
                mode=str(sample.mode),
                used_fallback=bool(sample.used_fallback),
                older_timestamp_s=sample.older_timestamp_s,
                newer_timestamp_s=sample.newer_timestamp_s,
                alpha=sample.alpha,
            )
        )

    aligned_window = ReferenceWindow(
        base_time_s=float(reference_window.base_time_s),
        policy_dt_s=float(reference_window.policy_dt_s),
        reference_steps=tuple(reference_window.reference_steps),
        samples=tuple(aligned_samples),
    )
    return aligned_window, fixed_yaw_quat, fixed_pivot_pos


def compute_anchor_velocities(
    obs_builder_base: object,
    qpos: Float64Array,
    last_reference_qpos: Float64Array | None,
    num_actions: int,
    policy_hz: float,
) -> tuple[Float32Array, Float32Array]:
    """Compute motion anchor linear/angular velocity in world frame via finite diff.

    Uses forward kinematics on obs_builder_base to get anchor body positions
    and quaternions, then computes velocities from the finite difference
    between current and previous reference qpos.
    """
    builder = obs_builder_base

    cur_pos = np.asarray(qpos[0:3], dtype=np.float32)
    cur_quat = np.asarray(qpos[3:7], dtype=np.float32)
    cur_joints = np.asarray(qpos[7:7 + num_actions], dtype=np.float32)
    builder._run_fk(cur_pos, cur_quat, cur_joints)
    cur_anchor_pos = builder._get_body_pos(builder._anchor_body_id).copy()

    if last_reference_qpos is None:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    prev = last_reference_qpos
    prev_pos = np.asarray(prev[0:3], dtype=np.float32)
    prev_quat = np.asarray(prev[3:7], dtype=np.float32)
    prev_joints = np.asarray(prev[7:7 + num_actions], dtype=np.float32)
    builder._run_fk(prev_pos, prev_quat, prev_joints)
    prev_anchor_pos = builder._get_body_pos(builder._anchor_body_id).copy()

    dt = np.float32(1.0 / policy_hz)
    anchor_lin_vel_w = np.asarray(
        (cur_anchor_pos - prev_anchor_pos) / dt, dtype=np.float32,
    )

    # Angular velocity from quaternion difference.
    builder._run_fk(cur_pos, cur_quat, cur_joints)
    cur_anchor_quat = builder._get_body_quat(builder._anchor_body_id).copy()
    builder._run_fk(prev_pos, prev_quat, prev_joints)
    prev_anchor_quat = builder._get_body_quat(builder._anchor_body_id).copy()

    q_delta = quat_mul_np(cur_anchor_quat, quat_inv_np(prev_anchor_quat))
    if q_delta[0] < 0:
        q_delta = -q_delta
    w_clamped = float(np.clip(q_delta[0], -1.0, 1.0))
    half_angle = np.float32(np.arccos(w_clamped))
    sin_half = np.float32(np.sin(half_angle))
    if sin_half > 1e-6:
        axis = q_delta[1:4] / sin_half
        anchor_ang_vel_w = np.asarray(axis * 2.0 * half_angle / dt, dtype=np.float32)
    else:
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)

    # NaN/inf safety guards
    _zero3 = np.zeros(3, dtype=np.float32)
    if not np.all(np.isfinite(anchor_lin_vel_w)):
        logger.warning("NaN/inf in anchor_lin_vel_w, damping to zero")
        anchor_lin_vel_w = _zero3
    if not np.all(np.isfinite(anchor_ang_vel_w)):
        logger.warning("NaN/inf in anchor_ang_vel_w, damping to zero")
        anchor_ang_vel_w = _zero3

    return anchor_lin_vel_w, anchor_ang_vel_w


def dispatch_build_observation(
    obs_builder: object,
    state: object,
    reference_window: ReferenceWindow | None,
    aligned_reference_window: ReferenceWindow | None,
    motion_qpos: Float32Array,
    motion_joint_vel: Float32Array,
    last_action: Float32Array,
    anchor_lin_vel_w: Float32Array | None,
    anchor_ang_vel_w: Float32Array | None,
) -> Float32Array:
    """Build observation, dispatching to reference-window or velocity-command path.

    If obs_builder supports SupportsReferenceWindow, uses aligned_reference_window.
    Otherwise uses anchor velocities.
    """
    if isinstance(obs_builder, SupportsReferenceWindow):
        obs = obs_builder.build_with_reference_window(
            state,
            aligned_reference_window,
            motion_qpos,
            last_action,
        )
    else:
        assert anchor_lin_vel_w is not None
        assert anchor_ang_vel_w is not None
        obs = obs_builder.build(
            state,
            motion_qpos,
            motion_joint_vel,
            last_action,
            anchor_lin_vel_w,
            anchor_ang_vel_w,
        )
    return np.asarray(obs, dtype=np.float32)
