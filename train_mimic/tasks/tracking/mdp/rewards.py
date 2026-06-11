from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
    quat_error_magnitude,
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


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str | tuple[str, ...],
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collision slots above the configured force threshold."""
    hit = _self_collision_hits(env, sensor_name, force_threshold)
    return hit.sum(dim=-1).float()


def _self_collision_hits(
    env: ManagerBasedRlEnv,
    sensor_name: str | tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    sensor_names = (sensor_name,) if isinstance(sensor_name, str) else sensor_name
    force_histories = []
    found_values = []
    for name in sensor_names:
        data = env.scene[name].data
        if data.force_history is not None:
            force_histories.append(data.force_history)
        else:
            assert data.found is not None
            found = data.found
            if found.ndim == 1:
                found = found.unsqueeze(-1)
            found_values.append(found)

    if force_histories:
        force_history = torch.cat(force_histories, dim=1)
        force_mag = torch.norm(force_history, dim=-1)
        return (force_mag > force_threshold).any(dim=2)

    found = torch.cat(found_values, dim=1)
    return (found > 0).any(dim=1, keepdim=True)
