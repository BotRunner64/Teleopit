from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, cast, final

import numpy as np

from teleopit.interfaces import RobotState

FloatVec = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
ConfigType = dict[str, object] | object
_GRAVITY_UNIT_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_CFG_MISSING = object()


def _as_float_vec(value: object, name: str) -> FloatVec:
    out = np.asarray(value, dtype=np.float32)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return out


def _cfg_get(cfg: ConfigType, key: str, default: object = _CFG_MISSING) -> object:
    if isinstance(cfg, dict):
        if key in cfg:
            return cast(object, cfg[key])
        if default is not _CFG_MISSING:
            return default
        raise KeyError(key)
    if hasattr(cfg, key):
        return cast(object, object.__getattribute__(cfg, key))
    if default is not _CFG_MISSING:
        return default
    raise KeyError(key)


def _as_int_scalar(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _quat_inv_np(q: FloatVec) -> FloatVec:
    inv = q.copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def _quat_mul_np(q1: FloatVec, q2: FloatVec) -> FloatVec:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1).astype(np.float32)


def _quat_rotate_np(q: FloatVec, v: FloatVec) -> FloatVec:
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=np.float32)
    v_quat[..., 1:4] = v
    result = _quat_mul_np(_quat_mul_np(q, v_quat), _quat_inv_np(q))
    return result[..., 1:4]


def _yaw_quat_np(q: FloatVec) -> FloatVec:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)
    out /= max(float(np.linalg.norm(out)), 1e-8)
    return out


def align_motion_qpos_yaw(
    robot_quat_wxyz: FloatVec,
    motion_qpos: np.ndarray,
) -> np.ndarray:
    motion_quat = np.asarray(motion_qpos[3:7], dtype=np.float32)
    robot_quat = np.asarray(robot_quat_wxyz, dtype=np.float32)
    delta = _quat_mul_np(robot_quat, _quat_inv_np(motion_quat))
    delta_yaw = _yaw_quat_np(delta)
    motion_qpos[3:7] = _quat_mul_np(delta_yaw, motion_quat).astype(motion_qpos.dtype)
    return motion_qpos


def compute_fixed_yaw_alignment_quat(
    robot_quat_wxyz: FloatVec,
    motion_quat_wxyz: FloatVec,
) -> FloatVec:
    robot_quat = np.asarray(robot_quat_wxyz, dtype=np.float32)
    motion_quat = np.asarray(motion_quat_wxyz, dtype=np.float32)
    delta = _quat_mul_np(robot_quat, _quat_inv_np(motion_quat))
    return _yaw_quat_np(delta)


def rotate_motion_qpos_by_yaw(
    motion_qpos: np.ndarray,
    yaw_offset_quat_wxyz: FloatVec,
    pivot_pos_w: FloatVec | None = None,
) -> np.ndarray:
    yaw_offset = np.asarray(yaw_offset_quat_wxyz, dtype=np.float32).reshape(4)
    pivot = (
        np.asarray(motion_qpos[0:3], dtype=np.float32)
        if pivot_pos_w is None
        else np.asarray(pivot_pos_w, dtype=np.float32).reshape(3)
    )
    base_pos = np.asarray(motion_qpos[0:3], dtype=np.float32)
    base_quat = np.asarray(motion_qpos[3:7], dtype=np.float32)

    rotated_pos = _quat_rotate_np(yaw_offset, base_pos - pivot) + pivot
    rotated_quat = _quat_mul_np(yaw_offset, base_quat)

    motion_qpos[0:3] = rotated_pos.astype(motion_qpos.dtype)
    motion_qpos[3:7] = rotated_quat.astype(motion_qpos.dtype)
    return motion_qpos


def _quat_to_rot6d_np(q: FloatVec) -> FloatVec:
    w, x, y, z = q[0], q[1], q[2], q[3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)

@final
class _VelCmdBaseObservationBuilder:
    """Internal base block used by the public 166D VelCmd builder."""

    def __init__(self, cfg: ConfigType) -> None:
        self.num_actions: int = _as_int_scalar(_cfg_get(cfg, "num_actions"), "num_actions")
        self.default_dof_pos: FloatVec = _as_float_vec(
            _cfg_get(cfg, "default_dof_pos"), "default_dof_pos"
        )
        if self.default_dof_pos.shape[0] != self.num_actions:
            raise ValueError("default_dof_pos length must match num_actions")

        xml_path = str(_cfg_get(cfg, "xml_path"))
        if not Path(xml_path).is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        import mujoco

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)
        self._body_name_to_idx: dict[str, int] = {
            self._mj_model.body(i).name: i for i in range(self._mj_model.nbody)
        }

        anchor_name = str(_cfg_get(cfg, "anchor_body_name"))
        if anchor_name not in self._body_name_to_idx:
            raise ValueError(f"Anchor body '{anchor_name}' not found in model")
        self._anchor_body_id = self._body_name_to_idx[anchor_name]
        self.total_obs_size = self.num_actions * 2 + 6 + 3 + self.num_actions * 3

    def reset(self) -> None:
        pass

    def _run_fk(self, base_pos: FloatVec, base_quat: FloatVec, joint_pos: FloatVec) -> None:
        import mujoco

        self._mj_data.qpos[:] = 0.0
        self._mj_data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        self._mj_data.qpos[3:7] = quat
        n = min(len(joint_pos), self._mj_model.nq - 7)
        self._mj_data.qpos[7:7 + n] = np.asarray(joint_pos, dtype=np.float64)[:n]
        mujoco.mj_kinematics(self._mj_model, self._mj_data)

    def _get_body_pos(self, body_id: int) -> FloatVec:
        return np.asarray(self._mj_data.xpos[body_id], dtype=np.float32).copy()

    def _get_body_quat(self, body_id: int) -> FloatVec:
        return np.asarray(self._mj_data.xquat[body_id], dtype=np.float32).copy()

    def build(
        self,
        robot_state: RobotState,
        motion_qpos: FloatVec,
        motion_joint_vel: FloatVec,
        last_action: FloatVec,
    ) -> FloatVec:
        qpos = np.asarray(robot_state.qpos, dtype=np.float32).reshape(-1)[: self.num_actions]
        qvel = np.asarray(robot_state.qvel, dtype=np.float32).reshape(-1)[: self.num_actions]
        robot_quat = np.asarray(robot_state.quat, dtype=np.float32).reshape(-1)
        base_ang_vel = np.asarray(robot_state.ang_vel, dtype=np.float32).reshape(-1)
        if robot_quat.shape[0] != 4:
            raise ValueError(f"robot_state.quat must be 4D (wxyz), got {robot_quat.shape[0]}")
        if base_ang_vel.shape[0] != 3:
            raise ValueError(f"robot_state.ang_vel must be 3D, got {base_ang_vel.shape[0]}")

        motion = np.asarray(motion_qpos, dtype=np.float32).reshape(-1)
        if motion.shape[0] < 7 + self.num_actions:
            raise ValueError(
                f"motion_qpos must contain 7D root + {self.num_actions}D joints, got {motion.shape[0]}"
            )
        motion_joint_vel_vec = np.asarray(motion_joint_vel, dtype=np.float32).reshape(-1)[: self.num_actions]
        if motion_joint_vel_vec.shape[0] != self.num_actions:
            raise ValueError(
                f"motion_joint_vel must be {self.num_actions}D, got {motion_joint_vel_vec.shape[0]}"
            )
        last_act = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if last_act.shape[0] != self.num_actions:
            raise ValueError(f"last_action length must be {self.num_actions}, got {last_act.shape[0]}")

        self._run_fk(np.zeros(3, dtype=np.float32), robot_quat, qpos)
        robot_anchor_quat = self._get_body_quat(self._anchor_body_id)

        motion_base_pos = motion[0:3]
        motion_base_quat = motion[3:7]
        motion_joint_pos = motion[7:7 + self.num_actions]
        self._run_fk(motion_base_pos, motion_base_quat, motion_joint_pos)
        motion_anchor_quat = self._get_body_quat(self._anchor_body_id)

        command = np.concatenate((motion_joint_pos, motion_joint_vel_vec), dtype=np.float32)
        rel_quat = _quat_mul_np(_quat_inv_np(robot_anchor_quat), motion_anchor_quat)
        motion_anchor_ori_b = _quat_to_rot6d_np(rel_quat)
        joint_pos_rel = qpos - self.default_dof_pos

        obs = np.concatenate([
            command,
            motion_anchor_ori_b,
            base_ang_vel,
            joint_pos_rel,
            qvel,
            last_act,
        ], dtype=np.float32)
        if obs.shape[0] != self.total_obs_size:
            raise ValueError(f"Expected {self.total_obs_size}D base observation, got {obs.shape[0]}")
        return obs


@final
class VelCmdObservationBuilder:
    """166D observation builder for the only supported VelCmdHistory policy."""

    def __init__(self, cfg: ConfigType) -> None:
        self._base = _VelCmdBaseObservationBuilder(cfg)
        self.num_actions = self._base.num_actions
        self.total_obs_size = self._base.total_obs_size + 12

    def reset(self) -> None:
        self._base.reset()

    def build(
        self,
        robot_state: RobotState,
        motion_qpos: FloatVec,
        motion_joint_vel: FloatVec,
        last_action: FloatVec,
        motion_anchor_lin_vel_w: FloatVec,
        motion_anchor_ang_vel_w: FloatVec,
    ) -> FloatVec:
        base_obs = self._base.build(robot_state, motion_qpos, motion_joint_vel, last_action)

        robot_quat = np.asarray(robot_state.quat, dtype=np.float32).reshape(-1)
        motion = np.asarray(motion_qpos, dtype=np.float32).reshape(-1)
        self._base._run_fk(motion[0:3], motion[3:7], motion[7:7 + self.num_actions])
        motion_anchor_quat = self._base._get_body_quat(self._base._anchor_body_id)

        qpos = np.asarray(robot_state.qpos, dtype=np.float32).reshape(-1)[: self.num_actions]
        robot_base_pos = np.zeros(3, dtype=np.float32)
        if robot_state.base_pos is not None:
            robot_base_pos = np.asarray(robot_state.base_pos, dtype=np.float32).reshape(-1)
        self._base._run_fk(robot_base_pos, robot_quat, qpos)
        robot_anchor_quat = self._base._get_body_quat(self._base._anchor_body_id)

        ref_lin_vel_w = np.asarray(motion_anchor_lin_vel_w, dtype=np.float32).reshape(3)
        ref_ang_vel_w = np.asarray(motion_anchor_ang_vel_w, dtype=np.float32).reshape(3)
        projected_gravity = _quat_rotate_np(_quat_inv_np(robot_quat), _GRAVITY_UNIT_W)
        robot_inv = _quat_inv_np(robot_anchor_quat)
        ref_base_lin_vel_b = _quat_rotate_np(robot_inv, ref_lin_vel_w)
        ref_base_ang_vel_b = _quat_rotate_np(robot_inv, ref_ang_vel_w)
        ref_projected_gravity_b = _quat_rotate_np(_quat_inv_np(motion_anchor_quat), _GRAVITY_UNIT_W)

        velcmd_obs = np.concatenate([
            projected_gravity,
            ref_base_lin_vel_b,
            ref_base_ang_vel_b,
            ref_projected_gravity_b,
        ], dtype=np.float32)
        obs = np.concatenate([base_obs, velcmd_obs], dtype=np.float32)
        if obs.shape[0] != self.total_obs_size:
            raise ValueError(f"Expected {self.total_obs_size}D velcmd observation, got {obs.shape[0]}")
        return obs

    def build_observation(
        self,
        state: RobotState,
        history: list[FloatVec],
        action_mimic: FloatVec,
    ) -> FloatVec:
        raise ValueError(
            "VelCmdObservationBuilder requires full motion data. "
            "Use build(robot_state, motion_qpos, motion_joint_vel, last_action, "
            "motion_anchor_lin_vel_w, motion_anchor_ang_vel_w)."
        )

