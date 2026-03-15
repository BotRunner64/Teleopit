from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
    command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.sum(
        torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
    )
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_pos_relative_w[:, body_indexes]
            - command.robot_body_pos_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, body_indexes]
            - command.robot_body_lin_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, body_indexes]
            - command.robot_body_ang_vel_w[:, body_indexes]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def _robot_body_local(command: MotionCommand, body_indexes: list[int]):
    """Compute robot body positions/orientations in robot anchor's yaw-only frame."""
    robot_yaw_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
    robot_yaw_inv_rep = robot_yaw_inv[:, None, :].expand(
        -1, len(body_indexes), -1
    )
    pos_b = quat_apply(
        robot_yaw_inv_rep,
        command.robot_body_pos_w[:, body_indexes]
        - command.robot_anchor_pos_w[:, None, :],
    )
    quat_b = quat_mul(robot_yaw_inv_rep, command.robot_body_quat_w[:, body_indexes])
    return pos_b, quat_b, robot_yaw_inv_rep


# -- Velocity-driven root tracking rewards ------------------------------------


def root_lin_vel_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    cmd_vel = command.anchor_lin_vel_heading[:, :2]
    robot_vel = command.robot_anchor_lin_vel_heading[:, :2]
    error = torch.sum(torch.square(cmd_vel - robot_vel), dim=-1)
    return torch.exp(-error / std**2)


def root_yaw_rate_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = (command.anchor_yaw_rate - command.robot_anchor_yaw_rate) ** 2
    return torch.exp(-error / std**2)


def root_height_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = (command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) ** 2
    return torch.exp(-error / std**2)


# -- Local-frame body tracking rewards ----------------------------------------


def motion_local_body_position_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    robot_pos_b, _, _ = _robot_body_local(command, body_indexes)
    error = torch.sum(
        torch.square(command.body_pos_b[:, body_indexes] - robot_pos_b), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_orientation_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    _, robot_quat_b, _ = _robot_body_local(command, body_indexes)
    error = quat_error_magnitude(command.body_quat_b[:, body_indexes], robot_quat_b) ** 2
    return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_linear_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    robot_yaw_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
    robot_yaw_inv_rep = robot_yaw_inv[:, None, :].expand(
        -1, len(body_indexes), -1
    )
    robot_vel_b = quat_apply(
        robot_yaw_inv_rep, command.robot_body_lin_vel_w[:, body_indexes]
    )
    error = torch.sum(
        torch.square(command.body_lin_vel_b[:, body_indexes] - robot_vel_b), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_angular_velocity_error_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    body_indexes = _get_body_indexes(command, body_names)
    robot_yaw_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
    robot_yaw_inv_rep = robot_yaw_inv[:, None, :].expand(
        -1, len(body_indexes), -1
    )
    robot_ang_vel_b = quat_apply(
        robot_yaw_inv_rep, command.robot_body_ang_vel_w[:, body_indexes]
    )
    error = torch.sum(
        torch.square(command.body_ang_vel_b[:, body_indexes] - robot_ang_vel_b), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        force_mag = torch.norm(data.force_history, dim=-1)
        hit = (force_mag > force_threshold).any(dim=1)
        return hit.sum(dim=-1).float()
    assert data.found is not None
    return data.found.squeeze(-1)
