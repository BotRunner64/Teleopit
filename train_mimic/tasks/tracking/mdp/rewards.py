from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    quat_error_magnitude,
)

from .commands import MotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
    command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
    return [
        i
        for i, name in enumerate(command.cfg.body_names)
        if (body_names is None) or (name in body_names)
    ]


def survival(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Match the sibling motion_tracking task's constant alive reward."""
    return torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def joint_pos_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Joint position tracking reward using MotionCommand interface."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.mean(torch.abs(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)


def joint_vel_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Joint velocity tracking reward using MotionCommand interface."""
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    error = torch.mean(torch.abs(command.joint_vel - command.robot_joint_vel), dim=-1)
    return torch.exp(-error / std**2)


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


class joint_torque_limits:
    """Penalize actuator-force limit violations with a configurable soft margin."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self.asset: Entity = env.scene[asset_cfg.name]
        self.soft_factor = float(cfg.params.get("soft_factor", 0.9))
        if not (0.0 < self.soft_factor <= 1.0):
            raise ValueError(f"soft_factor must be in (0, 1], got {self.soft_factor}")

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_names = list(self.asset.joint_names)
        else:
            joint_names = [self.asset.joint_names[idx] for idx in joint_ids]
        actuator_names = list(self.asset.actuator_names)
        name_to_actuator = {name: idx for idx, name in enumerate(actuator_names)}
        actuator_ids: list[int] = []
        for joint_name in joint_names:
            if joint_name not in name_to_actuator:
                raise RuntimeError(f"Actuator for joint '{joint_name}' not found.")
            actuator_ids.append(name_to_actuator[joint_name])
        self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)

        force_range = torch.as_tensor(
            env.sim.model.actuator_forcerange, device=env.device, dtype=torch.float32
        )
        if force_range.ndim == 2:
            force_range = force_range.unsqueeze(0).expand(env.num_envs, -1, -1)
        elif force_range.ndim != 3:
            raise RuntimeError(
                f"Unexpected actuator_forcerange shape: {tuple(force_range.shape)}"
            )
        ctrl_ids = torch.as_tensor(
            self.asset.indexing.ctrl_ids, device=env.device, dtype=torch.long
        )
        force_range = force_range.index_select(1, ctrl_ids)
        torque_limit = torch.maximum(force_range[..., 0].abs(), force_range[..., 1].abs())
        self._soft_limits = torque_limit.index_select(1, self._actuator_ids) * self.soft_factor

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        soft_factor: float = 0.9,
    ) -> torch.Tensor:
        del env, asset_cfg, soft_factor
        applied_torque = self.asset.data.actuator_force.index_select(1, self._actuator_ids)
        violation = (applied_torque.abs() / self._soft_limits - 1.0).clamp_min(0.0)
        return -violation.sum(dim=1)


class feet_air_time_ref:
    """Reward landing timing that matches the reference contact schedule."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = str(cfg.params["sensor_name"])
        self.command_name = str(cfg.params["command_name"])
        self.threshold = float(cfg.params.get("thres", 0.5))
        self._reward_time = torch.zeros((env.num_envs, 0), dtype=torch.float32, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._reward_time[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        thres: float = 0.5,
    ) -> torch.Tensor:
        del sensor_name, command_name, thres
        sensor: ContactSensor = env.scene[self.sensor_name]
        data = sensor.data
        if data.found is None:
            raise RuntimeError(f"Contact sensor '{self.sensor_name}' must expose 'found'")
        current_contact = data.found > 0
        first_contact = sensor.compute_first_contact(dt=env.step_dt)
        command = cast(object, env.command_manager.get_term(self.command_name))
        target_contact = torch.as_tensor(
            cast(object, command).feet_standing, device=env.device, dtype=torch.bool
        )
        if self._reward_time.shape != current_contact.shape:
            self._reward_time = torch.zeros_like(current_contact, dtype=torch.float32)

        contact_match = current_contact == target_contact
        self._reward_time = self._reward_time + torch.where(
            contact_match,
            torch.full_like(self._reward_time, env.step_dt),
            torch.full_like(self._reward_time, -env.step_dt),
        )
        reward = torch.sum(
            (self._reward_time - self.threshold).clamp_max(0.0) * first_contact.float(),
            dim=1,
        )
        self._reward_time = self._reward_time * (~current_contact).float()
        return reward


class feet_air_time_ref_dense:
    """Dense contact/height shaping against the reference foot contact state."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self.asset: Entity = env.scene[asset_cfg.name]
        body_names = cast(list[str], cfg.params["body_names"])
        body_ids, _ = self.asset.find_bodies(body_names, preserve_order=True)
        self._body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)

        site_names = cast(list[str] | None, cfg.params.get("site_names"))
        self._site_ids: torch.Tensor | None = None
        if site_names is not None:
            site_ids, _ = self.asset.find_sites(site_names, preserve_order=True)
            if len(site_ids) != len(body_ids):
                raise ValueError(
                    "site_names must match body_names length for feet_air_time_ref_dense"
                )
            self._site_ids = torch.tensor(site_ids, device=env.device, dtype=torch.long)
        else:
            body2_names = cast(list[str] | None, cfg.params.get("body2_names"))
            if body2_names is None:
                self._body2_ids = self._body_ids
            else:
                body2_ids, _ = self.asset.find_bodies(body2_names, preserve_order=True)
                if len(body2_ids) != len(body_ids):
                    raise ValueError(
                        "body2_names must match body_names length for feet_air_time_ref_dense"
                    )
                self._body2_ids = torch.tensor(body2_ids, device=env.device, dtype=torch.long)

        self.sensor_name = str(cfg.params["sensor_name"])
        self.command_name = str(cfg.params["command_name"])
        self.air_h_low = float(cfg.params.get("air_h_low", 0.035))
        self.air_h_high = float(cfg.params.get("air_h_high", 0.155))
        self.contact_h_low = float(cfg.params.get("contact_h_low", 0.035))
        self.contact_h_high = float(cfg.params.get("contact_h_high", 0.125))
        self.air_h_span = max(self.air_h_high - self.air_h_low, 1.0e-6)
        self.contact_h_span = max(self.contact_h_high - self.contact_h_low, 1.0e-6)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        body_names: tuple[str, ...],
        body2_names: tuple[str, ...] | None = None,
        site_names: tuple[str, ...] | None = None,
        air_h_low: float = 0.035,
        air_h_high: float = 0.155,
        contact_h_low: float = 0.035,
        contact_h_high: float = 0.125,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del (
            sensor_name,
            command_name,
            body_names,
            body2_names,
            site_names,
            air_h_low,
            air_h_high,
            contact_h_low,
            contact_h_high,
            asset_cfg,
        )
        sensor: ContactSensor = env.scene[self.sensor_name]
        data = sensor.data
        if data.found is None:
            raise RuntimeError(f"Contact sensor '{self.sensor_name}' must expose 'found'")
        current_contact = data.found > 0
        command = cast(object, env.command_manager.get_term(self.command_name))
        target_contact = torch.as_tensor(
            cast(object, command).feet_standing, device=env.device, dtype=torch.bool
        )

        mismatch = current_contact ^ target_contact
        both_air = (~current_contact) & (~target_contact)
        both_contact = current_contact & target_contact

        penalty = torch.zeros_like(target_contact, dtype=torch.float32)
        penalty[mismatch] = -1.0

        if self._site_ids is not None:
            foot_probe_height = self.asset.data.site_pos_w.index_select(1, self._site_ids)[..., 2]
            feet_height_air = foot_probe_height
            feet_height_contact = foot_probe_height
        else:
            feet_height_air = torch.minimum(
                self.asset.data.body_link_pos_w.index_select(1, self._body_ids)[..., 2],
                self.asset.data.body_link_pos_w.index_select(1, self._body2_ids)[..., 2],
            )
            feet_height_contact = torch.maximum(
                self.asset.data.body_link_pos_w.index_select(1, self._body_ids)[..., 2],
                self.asset.data.body_link_pos_w.index_select(1, self._body2_ids)[..., 2],
            )
        air_ratio = ((feet_height_air - self.air_h_low) / self.air_h_span).clamp(0.0, 1.0)
        penalty = torch.where(both_air, -(1.0 - air_ratio), penalty)

        contact_ratio = (
            (feet_height_contact - self.contact_h_low) / self.contact_h_span
        ).clamp(0.0, 1.0)
        penalty = torch.where(both_contact, -contact_ratio, penalty)
        return penalty.mean(dim=1)


# ---------------------------------------------------------------------------
# TWIST2-style feet rewards (adapted to mjlab API)
# ---------------------------------------------------------------------------


class feet_air_time:
    """Reward feet air time matching a target duration (TWIST2-style).

    Accumulates air time per foot; on first contact, rewards
    ``(air_time - target).clamp(max=0)`` so that too-short air phases are
    penalised. Only active when the reference root XY speed > 0.05 m/s.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = str(cfg.params["sensor_name"])
        self.command_name = str(cfg.params["command_name"])
        self.air_time_target = float(cfg.params.get("air_time_target", 0.5))
        self.feet_air_time = torch.zeros(
            (env.num_envs, 0), dtype=torch.float32, device=env.device
        )
        self._last_contacts = torch.zeros(
            (env.num_envs, 0), dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.feet_air_time[env_ids] = 0.0
        self._last_contacts[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str,
        air_time_target: float = 0.5,
    ) -> torch.Tensor:
        del sensor_name, command_name, air_time_target
        sensor: ContactSensor = env.scene[self.sensor_name]
        data = sensor.data
        if data.force is None:
            raise RuntimeError(f"Contact sensor '{self.sensor_name}' must expose 'force'")
        contact = data.force[..., 2].abs() > 5.0  # [B, N]  — TWIST2: fz > 5N

        # Lazy init on first call (shape unknown at __init__)
        if self.feet_air_time.shape != contact.shape:
            self.feet_air_time = torch.zeros_like(contact, dtype=torch.float32)
            self._last_contacts = torch.zeros_like(contact, dtype=torch.bool)

        # contact_filt = contact OR last_contacts (same as TWIST2)
        contact_filt = contact | self._last_contacts
        self._last_contacts = contact

        first_contact = (self.feet_air_time > 0.0) & contact_filt

        self.feet_air_time += env.step_dt
        air_time = (self.feet_air_time - self.air_time_target) * first_contact.float()
        air_time = air_time.clamp(max=0.0)
        self.feet_air_time *= ~contact_filt

        reward = air_time.sum(dim=1)

        # Gate by reference root XY speed > 0.05
        command = cast(MotionCommand, env.command_manager.get_term(self.command_name))
        ref_root_vxy = torch.norm(command.body_lin_vel_w[:, 0, :2], dim=-1)
        reward *= (ref_root_vxy > 0.05).float()

        return reward


def feet_stumble(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalise stumbling: lateral contact force > 4x vertical (TWIST2-style)."""
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' must expose 'force'")
    force = data.force  # [B, N, 3]
    lateral = torch.norm(force[..., :2], dim=-1)  # [B, N]
    vertical = torch.abs(force[..., 2])  # [B, N]
    stumble = torch.any(lateral > 4.0 * vertical, dim=1)
    return stumble.float()


def feet_contact_forces(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_contact_force: float = 350.0,
) -> torch.Tensor:
    """Penalise excessive vertical contact forces (TWIST2-style).

    Computes L2 norm of vertical forces across all feet, then penalises the
    excess above *max_contact_force*.  This matches TWIST2's implementation
    where two feet at 300 N each (norm ≈ 424 N) would trigger a penalty.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' must expose 'force'")
    fz = data.force[..., 2]  # [B, N]
    fz_norm = torch.norm(fz, dim=-1)  # [B]
    excess = (fz_norm - max_contact_force).clamp(min=0.0)
    return excess


class feet_slip:
    """Penalise horizontal foot velocity while in contact (TWIST2-style)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg = cast(SceneEntityCfg, cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG))
        self.asset: Entity = env.scene[asset_cfg.name]
        body_names = cast(list[str], cfg.params["body_names"])
        body_ids, _ = self.asset.find_bodies(body_names, preserve_order=True)
        self._body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
        self.sensor_name = str(cfg.params["sensor_name"])

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        body_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del sensor_name, body_names, asset_cfg
        sensor: ContactSensor = env.scene[self.sensor_name]
        data = sensor.data
        if data.force is None:
            raise RuntimeError(f"Contact sensor '{self.sensor_name}' must expose 'force'")
        contact = (data.force[..., 2].abs() > 5.0).float()  # [B, N]  — TWIST2: fz > 5N

        feet_vel_xy = self.asset.data.body_link_lin_vel_w.index_select(
            1, self._body_ids
        )[..., :2]  # [B, N, 2]
        speed_xy = torch.norm(feet_vel_xy, dim=-1)  # [B, N]
        slip = torch.sqrt(speed_xy) * contact
        return slip.sum(dim=1)
