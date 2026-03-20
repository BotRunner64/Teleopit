from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    sample_uniform,
)

from train_mimic.data.dataset_lib import parse_window_steps

from .commands import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


FloatTensor = torch.Tensor


def _expand_quat(quat: FloatTensor, count: int) -> FloatTensor:
    return quat.unsqueeze(1).expand(-1, count, -1)


def _as_body_rot6d(quat: FloatTensor) -> FloatTensor:
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(mat.shape[0], mat.shape[1], -1)


def _exp_reward(error: FloatTensor, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    if isinstance(sigma, (int, float)):
        sigma_values = [float(sigma)]
    else:
        sigma_values = [float(v) for v in sigma]
    if not sigma_values:
        raise ValueError("sigma must be non-empty")
    rewards = [torch.exp(-error / max(value * value, 1.0e-6)) for value in sigma_values]
    reward = sum(rewards) / float(len(rewards))
    if reward.ndim > 1 and reward.shape[-1] == 1:
        return reward.squeeze(-1)
    return reward



def _as_body_rot6d_deploy(quat: FloatTensor) -> FloatTensor:
    mat = matrix_from_quat(quat)
    return mat[..., :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)


def motion_tracking_boot_indicator(env: "ManagerBasedRlEnv") -> FloatTensor:
    return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)


def motion_tracking_projected_gravity(
    env: "ManagerBasedRlEnv",
    entity_name: str = "robot",
) -> FloatTensor:
    asset: Entity = env.scene[entity_name]
    return quat_apply(quat_inv(asset.data.root_link_quat_w), asset.data.gravity_vec_w)


def motion_tracking_joint_pos_abs(
    env: "ManagerBasedRlEnv",
    entity_name: str = "robot",
) -> FloatTensor:
    asset: Entity = env.scene[entity_name]
    return asset.data.joint_pos


def motion_tracking_joint_vel_abs(
    env: "ManagerBasedRlEnv",
    entity_name: str = "robot",
) -> FloatTensor:
    asset: Entity = env.scene[entity_name]
    return asset.data.joint_vel


class MotionTrackingDeployHistoryTerm:
    def __init__(self, cfg: object, env: "ManagerBasedRlEnv"):
        params = cast(dict[str, object], getattr(cfg, "params"))
        source = str(params.get("source", "")).strip()
        if source not in {"root_ang_vel_b", "projected_gravity_b", "joint_pos", "joint_vel"}:
            raise ValueError(f"Unsupported deploy history source: {source}")
        steps_raw = params.get("steps")
        if not isinstance(steps_raw, (tuple, list)):
            raise ValueError("MotionTrackingDeployHistoryTerm.steps must be a sequence of ints")
        self.steps = tuple(int(v) for v in steps_raw)
        if not self.steps or self.steps[0] != 0 or min(self.steps) < 0:
            raise ValueError(f"Deploy history steps must be non-empty, non-negative, and start with 0, got {self.steps}")
        self.source = source
        self.device = env.device
        self.num_envs = env.num_envs
        self.max_step = max(self.steps)
        self._last_step = -1
        dim = self._source(env).shape[1]
        self.buffer = torch.zeros((self.num_envs, self.max_step + 1, dim), dtype=torch.float32, device=self.device)
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self.buffer.zero_()
            self.initialized[:] = False
            self._last_step = -1
            return
        self.buffer[env_ids] = 0.0
        self.initialized[env_ids] = False

    def _source(self, env: "ManagerBasedRlEnv") -> FloatTensor:
        asset: Entity = env.scene["robot"]
        if self.source == "root_ang_vel_b":
            return asset.data.root_link_ang_vel_b
        if self.source == "projected_gravity_b":
            return quat_apply(quat_inv(asset.data.root_link_quat_w), asset.data.gravity_vec_w)
        if self.source == "joint_pos":
            return asset.data.joint_pos
        return asset.data.joint_vel

    def __call__(self, env: "ManagerBasedRlEnv", **_: object) -> FloatTensor:
        current = self._source(env).to(torch.float32)
        current_step = int(env.common_step_counter)
        if current_step != self._last_step:
            if bool(self.initialized.any()):
                self.buffer[:, 1:] = self.buffer[:, :-1].clone()
            self.buffer[:, 0] = current
            self._last_step = current_step
        else:
            self.buffer[:, 0] = current

        if bool((~self.initialized).any()):
            env_ids = ~self.initialized
            self.buffer[env_ids] = current[env_ids].unsqueeze(1).expand(-1, self.max_step + 1, -1)
            self.initialized[env_ids] = True
        return self.buffer[:, self.steps].reshape(self.num_envs, -1)


class MotionTrackingPrevActionHistoryTerm:
    def __init__(self, cfg: object, env: "ManagerBasedRlEnv"):
        params = cast(dict[str, object], getattr(cfg, "params"))
        steps = int(params.get("steps", 0))
        if steps <= 0:
            raise ValueError(f"MotionTrackingPrevActionHistoryTerm.steps must be > 0, got {steps}")
        self.steps = steps
        self.device = env.device
        self.num_envs = env.num_envs
        self.action_dim = int(env.action_manager.total_action_dim)
        self._last_step = -1
        self.buffer = torch.zeros((self.num_envs, self.steps, self.action_dim), dtype=torch.float32, device=self.device)
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self.buffer.zero_()
            self.initialized[:] = False
            self._last_step = -1
            return
        self.buffer[env_ids] = 0.0
        self.initialized[env_ids] = False

    def __call__(self, env: "ManagerBasedRlEnv", **_: object) -> FloatTensor:
        current = env.action_manager.action.to(torch.float32)
        current_step = int(env.common_step_counter)
        if current_step != self._last_step:
            if bool(self.initialized.any()):
                self.buffer[:, 1:] = self.buffer[:, :-1].clone()
            self.buffer[:, 0] = current
            self._last_step = current_step
        else:
            self.buffer[:, 0] = current

        if bool((~self.initialized).any()):
            env_ids = ~self.initialized
            self.buffer[env_ids] = 0.0
            self.initialized[env_ids] = True
        return self.buffer.reshape(self.num_envs, -1)


class MotionTrackingCommand(MotionCommand):
    cfg: "MotionTrackingCommandCfg"

    def __init__(self, cfg: "MotionTrackingCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._window_steps = parse_window_steps(cfg.window_steps)
        self._window_step_tensor = torch.tensor(self._window_steps, dtype=torch.float32, device=self.device)

        self.root_body_index = self.cfg.body_names.index(self.cfg.root_body_name)
        self.keypoint_body_indexes = self._resolve_body_indexes(self.cfg.keypoint_body_names)
        self.lower_keypoint_body_indexes = self._resolve_body_indexes(self.cfg.lower_keypoint_body_names)
        self.upper_keypoint_body_indexes = self._resolve_body_indexes(self.cfg.upper_keypoint_body_names)
        self.feet_body_indexes = self._resolve_body_indexes(self.cfg.feet_body_names)
        self.body_z_body_indexes = self._resolve_body_indexes(self.cfg.body_z_terminate_body_names)
        self.tracking_joint_indexes = self._resolve_joint_indexes(self.cfg.tracking_joint_names)

        num_tracking_joints = len(self.tracking_joint_indexes)
        self._target_joint_pos_bias = torch.zeros(
            (self.num_envs, num_tracking_joints), dtype=torch.float32, device=self.device
        )
        self._root_drift_vel_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._root_z_offset = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.feet_standing = torch.zeros(
            (self.num_envs, len(self.feet_body_indexes)), dtype=torch.float32, device=self.device
        )
        self._window_frames: dict[str, FloatTensor] = {}
        self._refresh_window_cache()
        self._update_feet_standing()

        self.metrics["tracking_root_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_keypoint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_joint_pos"] = torch.zeros(self.num_envs, device=self.device)

    def _resolve_body_indexes(self, body_names: tuple[str, ...]) -> list[int]:
        indexes: list[int] = []
        for name in body_names:
            if name not in self.cfg.body_names:
                raise ValueError(f"Body '{name}' is not present in motion/body_names")
            indexes.append(self.cfg.body_names.index(name))
        return indexes

    def _resolve_joint_indexes(self, joint_names: tuple[str, ...]) -> list[int]:
        indexes: list[int] = []
        robot_joint_names = list(self.robot.joint_names)
        for name in joint_names:
            if name not in robot_joint_names:
                raise ValueError(f"Joint '{name}' is not present in robot.joint_names")
            indexes.append(robot_joint_names.index(name))
        return indexes

    @property
    def reference_steps(self) -> tuple[int, ...]:
        return self._window_steps

    @property
    def target_joint_pos_window(self) -> FloatTensor:
        joint_pos = self._window_frames["joint_pos"][:, :, self.tracking_joint_indexes]
        return joint_pos + self._target_joint_pos_bias.unsqueeze(1)

    @property
    def target_joint_vel_window(self) -> FloatTensor:
        return self._window_frames["joint_vel"][:, :, self.tracking_joint_indexes]

    @property
    def target_body_pos_w_window(self) -> FloatTensor:
        pos = self._window_frames["body_pos_w"] + self._env.scene.env_origins[:, None, None, :]
        return pos + self._drift_offsets().unsqueeze(2)

    @property
    def target_body_quat_w_window(self) -> FloatTensor:
        return self._window_frames["body_quat_w"]

    @property
    def target_body_lin_vel_w_window(self) -> FloatTensor:
        vel = self._window_frames["body_lin_vel_w"].clone()
        vel = vel + self._root_drift_vel_w[:, None, None, :]
        return vel

    @property
    def target_body_ang_vel_w_window(self) -> FloatTensor:
        return self._window_frames["body_ang_vel_w"]

    @property
    def target_root_pos_w_window(self) -> FloatTensor:
        return self.target_body_pos_w_window[:, :, self.root_body_index]

    @property
    def target_root_quat_w_window(self) -> FloatTensor:
        return self.target_body_quat_w_window[:, :, self.root_body_index]

    @property
    def target_root_lin_vel_w_window(self) -> FloatTensor:
        return self.target_body_lin_vel_w_window[:, :, self.root_body_index]

    @property
    def target_root_ang_vel_w_window(self) -> FloatTensor:
        return self.target_body_ang_vel_w_window[:, :, self.root_body_index]

    @property
    def current_target_keypoint_pos_w(self) -> FloatTensor:
        return self.target_body_pos_w_window[:, 0, self.keypoint_body_indexes]

    @property
    def current_target_keypoint_quat_w(self) -> FloatTensor:
        return self.target_body_quat_w_window[:, 0, self.keypoint_body_indexes]

    @property
    def current_target_keypoint_lin_vel_w(self) -> FloatTensor:
        return self.target_body_lin_vel_w_window[:, 0, self.keypoint_body_indexes]

    @property
    def current_target_keypoint_ang_vel_w(self) -> FloatTensor:
        return self.target_body_ang_vel_w_window[:, 0, self.keypoint_body_indexes]

    def _drift_offsets(self) -> FloatTensor:
        offsets = self._window_step_tensor.view(1, -1, 1) * float(self._step_dt)
        drift = self._root_drift_vel_w.unsqueeze(1) * offsets
        drift[..., 2] += self._root_z_offset.unsqueeze(1)
        return drift

    def _sample_target_randomization(self, env_ids: FloatTensor) -> None:
        if env_ids.numel() == 0:
            return
        n = int(env_ids.numel())

        self._target_joint_pos_bias[env_ids] = sample_uniform(
            self.cfg.target_joint_pos_bias_range[0],
            self.cfg.target_joint_pos_bias_range[1],
            (n, self._target_joint_pos_bias.shape[1]),
            device=self.device,
        )
        self._root_drift_vel_w[env_ids] = 0.0
        if self.cfg.root_drift_vel_xy_max > 0.0:
            self._root_drift_vel_w[env_ids, 0:2] = sample_uniform(
                -self.cfg.root_drift_vel_xy_max,
                self.cfg.root_drift_vel_xy_max,
                (n, 2),
                device=self.device,
            )
        if self.cfg.root_drift_vel_z_max > 0.0:
            self._root_drift_vel_w[env_ids, 2] = sample_uniform(
                -self.cfg.root_drift_vel_z_max,
                self.cfg.root_drift_vel_z_max,
                (n,),
                device=self.device,
            )
        self._root_z_offset[env_ids] = sample_uniform(
            self.cfg.root_z_offset_range[0],
            self.cfg.root_z_offset_range[1],
            (n,),
            device=self.device,
        )

    def _refresh_window_cache(self) -> None:
        self._window_frames = self.motion.get_window_frames(
            self.motion_ids,
            self.motion_times,
            window_steps=self._window_steps,
        )

    def _update_feet_standing(self) -> None:
        if not self.feet_body_indexes:
            return
        feet_pos_w = self.target_body_pos_w_window[:, 0, self.feet_body_indexes]
        feet_vel_w = self.target_body_lin_vel_w_window[:, 0, self.feet_body_indexes]
        root_vxy = torch.norm(self.target_root_lin_vel_w_window[:, 0, :2], dim=-1, keepdim=True).clamp_min(1.0)
        feet_vxy = torch.norm(feet_vel_w[..., :2], dim=-1)
        feet_vz = feet_vel_w[..., 2].abs()
        feet_z = feet_pos_w[..., 2]
        standing = (
            (feet_z < self.cfg.feet_standing_z_threshold)
            & (feet_vxy < self.cfg.feet_standing_vxy_threshold * root_vxy)
            & (feet_vz < self.cfg.feet_standing_vz_threshold * root_vxy)
        )
        self.feet_standing = standing.float()

    def _resample_command(self, env_ids: FloatTensor):
        super()._resample_command(env_ids)
        if env_ids.numel() == 0:
            return
        self._sample_target_randomization(env_ids)
        self._refresh_window_cache()
        self._update_feet_standing()

    def _update_command(self):
        super()._update_command()
        self._refresh_window_cache()
        self._update_feet_standing()
        self._update_tracking_metrics()

    def _update_tracking_metrics(self) -> None:
        root_pos_error = torch.norm(
            self.target_root_pos_w_window[:, 0] - self.robot.data.root_link_pos_w,
            dim=-1,
        )
        self.metrics["tracking_root_pos"] = root_pos_error

        if self.keypoint_body_indexes:
            keypoint_error = torch.norm(
                self.current_target_keypoint_pos_w - self.robot_body_pos_w[:, self.keypoint_body_indexes],
                dim=-1,
            ).mean(dim=-1)
            self.metrics["tracking_keypoint_pos"] = keypoint_error

        if self.tracking_joint_indexes:
            joint_error = torch.norm(
                self.target_joint_pos_window[:, 0] - self.robot_joint_pos[:, self.tracking_joint_indexes],
                dim=-1,
            )
            self.metrics["tracking_joint_pos"] = joint_error


@dataclass(kw_only=True)
class MotionTrackingCommandCfg(MotionCommandCfg):
    root_body_name: str = "pelvis"
    keypoint_body_names: tuple[str, ...] = field(default_factory=tuple)
    lower_keypoint_body_names: tuple[str, ...] = field(default_factory=tuple)
    upper_keypoint_body_names: tuple[str, ...] = field(default_factory=tuple)
    feet_body_names: tuple[str, ...] = field(default_factory=tuple)
    body_z_terminate_body_names: tuple[str, ...] = field(default_factory=tuple)
    tracking_joint_names: tuple[str, ...] = field(default_factory=tuple)
    target_joint_pos_bias_range: tuple[float, float] = (-0.1, 0.1)
    root_drift_vel_xy_max: float = 0.25
    root_drift_vel_z_max: float = 0.05
    root_z_offset_range: tuple[float, float] = (-0.03, 0.03)
    feet_standing_z_threshold: float = 0.18
    feet_standing_vxy_threshold: float = 0.2
    feet_standing_vz_threshold: float = 0.15

    def build(self, env: "ManagerBasedRlEnv") -> MotionTrackingCommand:
        return MotionTrackingCommand(self, env)


def _get_tracking_command(env: "ManagerBasedRlEnv", command_name: str) -> MotionTrackingCommand:
    return cast(MotionTrackingCommand, env.command_manager.get_term(command_name))


def motion_tracking_command_obs(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    root_pos = asset.data.root_link_pos_w
    root_quat = asset.data.root_link_quat_w

    target_pos = command.target_root_pos_w_window
    target_quat = command.target_root_quat_w_window
    num_steps = target_pos.shape[1]
    root_inv = _expand_quat(quat_inv(root_quat), num_steps)
    rel_pos = quat_apply(root_inv, target_pos - root_pos.unsqueeze(1))
    rel_quat = quat_mul(root_inv, target_quat)
    rel_rot6d = _as_body_rot6d(rel_quat)
    return torch.cat(
        [rel_pos.reshape(env.num_envs, -1), rel_rot6d.reshape(env.num_envs, -1)],
        dim=-1,
    )


def motion_tracking_target_joint_pos(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return command.target_joint_pos_window.reshape(env.num_envs, -1)


def motion_tracking_target_root_height(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return command.target_root_pos_w_window[..., 2:3].reshape(env.num_envs, -1)


def motion_tracking_target_projected_gravity(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    gravity = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, len(command.reference_steps), -1)
    projected = quat_apply(quat_inv(command.target_root_quat_w_window), gravity)
    return projected.reshape(env.num_envs, -1)


def motion_tracking_target_feet_contact_state(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return command.feet_standing



def motion_tracking_deploy_command_obs(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    target_pos = command.target_root_pos_w_window
    target_quat = command.target_root_quat_w_window
    num_steps = target_pos.shape[1]
    if num_steps == 0:
        raise ValueError("MotionTrackingDeploy command window must contain at least one step")

    if num_steps > 1:
        pos_diff_w = target_pos[:, 1:] - target_pos[:, 0:1]
        ref_inv = _expand_quat(quat_inv(target_quat[:, 0]), num_steps - 1)
        pos_diff_b = quat_apply(ref_inv, pos_diff_w).reshape(env.num_envs, -1)
    else:
        pos_diff_b = torch.zeros((env.num_envs, 0), dtype=torch.float32, device=command.device)

    robot_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), num_steps)
    rel_quat = quat_mul(robot_inv, target_quat)
    rel_rot6d = _as_body_rot6d_deploy(rel_quat).reshape(env.num_envs, -1)
    return torch.cat([pos_diff_b, rel_rot6d], dim=-1)


def motion_tracking_deploy_target_joint_pos(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    target_joint_pos = command.target_joint_pos_window
    current_joint_pos = command.robot_joint_pos[:, command.tracking_joint_indexes].unsqueeze(1)
    return torch.cat(
        [
            target_joint_pos.reshape(env.num_envs, -1),
            (target_joint_pos - current_joint_pos).reshape(env.num_envs, -1),
        ],
        dim=-1,
    )


def motion_tracking_deploy_target_root_z(
    env: "ManagerBasedRlEnv",
    command_name: str,
    offset: float = 0.035,
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return (command.target_root_pos_w_window[..., 2] + float(offset)).reshape(env.num_envs, -1)


def motion_tracking_deploy_target_projected_gravity(
    env: "ManagerBasedRlEnv",
    command_name: str,
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    gravity = asset.data.gravity_vec_w.unsqueeze(1).expand(-1, len(command.reference_steps), -1)
    projected = quat_apply(quat_inv(command.target_root_quat_w_window), gravity)
    return projected.reshape(env.num_envs, -1)


def motion_tracking_current_keypoint_pos_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_pos = asset.data.root_link_pos_w.unsqueeze(1)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel = quat_apply(root_inv, command.robot_body_pos_w[:, command.keypoint_body_indexes] - root_pos)
    return rel.reshape(env.num_envs, -1)


def motion_tracking_target_keypoint_pos_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_pos = asset.data.root_link_pos_w.unsqueeze(1)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel = quat_apply(root_inv, command.current_target_keypoint_pos_w - root_pos)
    return rel.reshape(env.num_envs, -1)


def motion_tracking_current_keypoint_rot_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel_quat = quat_mul(root_inv, command.robot_body_quat_w[:, command.keypoint_body_indexes])
    return _as_body_rot6d(rel_quat).reshape(env.num_envs, -1)


def motion_tracking_target_keypoint_rot_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel_quat = quat_mul(root_inv, command.current_target_keypoint_quat_w)
    return _as_body_rot6d(rel_quat).reshape(env.num_envs, -1)


def motion_tracking_current_keypoint_linvel_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel = command.robot_body_lin_vel_w[:, command.keypoint_body_indexes] - asset.data.root_link_lin_vel_w.unsqueeze(1)
    return quat_apply(root_inv, rel).reshape(env.num_envs, -1)


def motion_tracking_target_keypoint_linvel_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel = command.current_target_keypoint_lin_vel_w - command.target_root_lin_vel_w_window[:, 0].unsqueeze(1)
    return quat_apply(root_inv, rel).reshape(env.num_envs, -1)


def motion_tracking_current_keypoint_angvel_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel = command.robot_body_ang_vel_w[:, command.keypoint_body_indexes] - asset.data.root_link_ang_vel_w.unsqueeze(1)
    return quat_apply(root_inv, rel).reshape(env.num_envs, -1)


def motion_tracking_target_keypoint_angvel_b(env: "ManagerBasedRlEnv", command_name: str) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    if not command.keypoint_body_indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    root_inv = _expand_quat(quat_inv(asset.data.root_link_quat_w), len(command.keypoint_body_indexes))
    rel = command.current_target_keypoint_ang_vel_w - command.target_root_ang_vel_w_window[:, 0].unsqueeze(1)
    return quat_apply(root_inv, rel).reshape(env.num_envs, -1)


def motion_tracking_body_height_obs(
    env: "ManagerBasedRlEnv", command_name: str, body_names: tuple[str, ...] | None = None
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    indexes = command.body_z_body_indexes if body_names is None else command._resolve_body_indexes(body_names)
    if not indexes:
        return torch.zeros((env.num_envs, 0), device=command.device)
    return command.robot_body_pos_w[:, indexes, 2]


def motion_tracking_root_pos_tracking(
    env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    error = torch.norm(command.target_root_pos_w_window[:, 0] - command.robot.data.root_link_pos_w, dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_root_rot_tracking(
    env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    error = quat_error_magnitude(command.target_root_quat_w_window[:, 0], command.robot.data.root_link_quat_w).unsqueeze(-1)
    return _exp_reward(error, sigma)


def motion_tracking_root_vel_tracking(
    env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    error = torch.norm(
        command.target_root_lin_vel_w_window[:, 0] - command.robot.data.root_link_lin_vel_w,
        dim=-1,
        keepdim=True,
    )
    return _exp_reward(error, sigma)


def motion_tracking_root_ang_vel_tracking(
    env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    error = torch.norm(
        command.target_root_ang_vel_w_window[:, 0] - command.robot.data.root_link_ang_vel_w,
        dim=-1,
        keepdim=True,
    )
    return _exp_reward(error, sigma)


def _keypoint_tracking_reward(
    env: "ManagerBasedRlEnv",
    command_name: str,
    indexes: list[int],
    sigma: float | list[float] | tuple[float, ...],
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    if not indexes:
        return torch.ones((env.num_envs,), device=command.device)
    error = torch.norm(
        command.target_body_pos_w_window[:, 0, indexes] - command.robot_body_pos_w[:, indexes],
        dim=-1,
    ).mean(dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_keypoint_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return _keypoint_tracking_reward(env, command_name, command.keypoint_body_indexes, sigma)


def motion_tracking_lower_keypoint_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return _keypoint_tracking_reward(env, command_name, command.lower_keypoint_body_indexes, sigma)


def motion_tracking_upper_keypoint_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    return _keypoint_tracking_reward(env, command_name, command.upper_keypoint_body_indexes, sigma)


def motion_tracking_keypoint_vel_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    if not command.keypoint_body_indexes:
        return torch.ones((env.num_envs,), device=command.device)
    error = torch.norm(
        command.target_body_lin_vel_w_window[:, 0, command.keypoint_body_indexes]
        - command.robot_body_lin_vel_w[:, command.keypoint_body_indexes],
        dim=-1,
    ).mean(dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_keypoint_rot_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    if not command.keypoint_body_indexes:
        return torch.ones((env.num_envs,), device=command.device)
    error = quat_error_magnitude(
        command.target_body_quat_w_window[:, 0, command.keypoint_body_indexes],
        command.robot_body_quat_w[:, command.keypoint_body_indexes],
    ).mean(dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_keypoint_angvel_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    if not command.keypoint_body_indexes:
        return torch.ones((env.num_envs,), device=command.device)
    error = torch.norm(
        command.target_body_ang_vel_w_window[:, 0, command.keypoint_body_indexes]
        - command.robot_body_ang_vel_w[:, command.keypoint_body_indexes],
        dim=-1,
    ).mean(dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_joint_pos_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    if not command.tracking_joint_indexes:
        return torch.ones((env.num_envs,), device=command.device)
    error = torch.abs(
        command.target_joint_pos_window[:, 0] - command.robot_joint_pos[:, command.tracking_joint_indexes]
    ).mean(dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_joint_vel_tracking(env: "ManagerBasedRlEnv", command_name: str, sigma: float | list[float] | tuple[float, ...]) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    if not command.tracking_joint_indexes:
        return torch.ones((env.num_envs,), device=command.device)
    error = torch.abs(
        command.target_joint_vel_window[:, 0] - command.robot_joint_vel[:, command.tracking_joint_indexes]
    ).mean(dim=-1, keepdim=True)
    return _exp_reward(error, sigma)


def motion_tracking_body_z_failure(
    env: "ManagerBasedRlEnv",
    command_name: str,
    threshold: float,
    body_names: tuple[str, ...] | None = None,
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    indexes = command.body_z_body_indexes if body_names is None else command._resolve_body_indexes(body_names)
    if not indexes:
        return torch.zeros((env.num_envs,), dtype=torch.bool, device=command.device)
    error = torch.abs(
        command.target_body_pos_w_window[:, 0, indexes, 2] - command.robot_body_pos_w[:, indexes, 2]
    )
    return torch.any(error > threshold, dim=-1)


def motion_tracking_gravity_failure(
    env: "ManagerBasedRlEnv",
    command_name: str,
    threshold: float,
) -> FloatTensor:
    command = _get_tracking_command(env, command_name)
    asset: Entity = env.scene[command.cfg.entity_name]
    gravity = asset.data.gravity_vec_w
    target_g = quat_apply(quat_inv(command.target_root_quat_w_window[:, 0]), gravity)
    robot_g = quat_apply(quat_inv(command.robot.data.root_link_quat_w), gravity)
    error = torch.norm(target_g - robot_g, dim=-1)
    return error > threshold
