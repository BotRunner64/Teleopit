from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
    yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Velocity-command observation terms: reference velocities and projected
# gravity for the VelCmd task variant.
# ---------------------------------------------------------------------------


def ref_base_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor linear velocity in the robot's body frame. (N, 3)"""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return quat_apply(quat_inv(command.robot_anchor_quat_w), command.anchor_lin_vel_w)


def ref_base_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor angular velocity in the robot's body frame. (N, 3)"""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return quat_apply(quat_inv(command.robot_anchor_quat_w), command.anchor_ang_vel_w)


def ref_projected_gravity_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Gravity direction in the reference anchor's body frame. (N, 3)

    Encodes the reference body tilt — analogous to ``projected_gravity`` but
    for the motion reference rather than the robot.
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset = env.scene[command.cfg.entity_name]
    return quat_apply(quat_inv(command.anchor_quat_w), asset.data.gravity_vec_w)


def ref_base_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Reference anchor height (z-coordinate). (N, 1) — critic privileged."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    return command.anchor_pos_w[:, 2:3]


# ---------------------------------------------------------------------------
# Yaw-only variants: use yaw_quat(robot_anchor_quat_w) to decouple
# roll/pitch from the coordinate transform, matching the TWIST2 approach.
# ---------------------------------------------------------------------------


def motion_anchor_pos_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        yaw_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        yaw_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    robot_yaw = yaw_quat(command.robot_anchor_quat_w)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_yaw[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b_yaw(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))

    num_bodies = len(command.cfg.body_names)
    robot_yaw = yaw_quat(command.robot_anchor_quat_w)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        robot_yaw[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


# ---------------------------------------------------------------------------
# Pure-ref windowed observation term
# ---------------------------------------------------------------------------


def ref_window_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Pure-ref windowed observations: ref_projected_gravity + ref_joint_pos + ref_joint_vel.

    Returns (B, W, 61) — only quantities that depend solely on the reference
    trajectory (no robot state), suitable for future-frame encoding.
    """
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    asset = env.scene[command.cfg.entity_name]
    gravity_w = asset.data.gravity_vec_w  # (B, 3)

    # ref_projected_gravity_b for each window frame: (B, W, 3)
    win_quat = command.window_anchor_quat_w  # (B, W, 4)
    B, W, _ = win_quat.shape
    win_quat_inv = quat_inv(win_quat.reshape(-1, 4)).reshape(B, W, 4)
    win_proj_grav = quat_apply(
        win_quat_inv.reshape(-1, 4),
        gravity_w[:, None, :].expand(B, W, 3).reshape(-1, 3),
    ).reshape(B, W, 3)

    # ref joint pos/vel: (B, W, 29)
    win_joint_pos = command.window_joint_pos
    win_joint_vel = command.window_joint_vel

    return torch.cat([win_proj_grav, win_joint_pos, win_joint_vel], dim=-1)
