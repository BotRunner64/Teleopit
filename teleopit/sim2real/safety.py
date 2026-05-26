"""Safety manager for Sim2Real controller.

Encapsulates KP ramp, joint limits, and velocity safety checks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.runtime.common import cfg_get
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

Float32Array = NDArray[np.float32]
logger = logging.getLogger(__name__)


class Sim2RealSafetyManager:
    """KP ramp, joint limits, and velocity safety checks."""

    def __init__(
        self,
        cfg: Any,
        robot: UnitreeG1Robot,
        policy_hz: float,
        num_actions: int,
    ) -> None:
        self._robot = robot
        self._policy_hz = policy_hz
        real_cfg = cfg_get(cfg, "real_robot")

        # KP ramp (gradually increase PD gains after episode-reset)
        _legacy_ramp_dur = cfg_get(cfg, "startup_ramp_duration", cfg_get(real_cfg, "startup_ramp_duration", 2.0))
        kp_ramp_dur = float(cfg_get(cfg, "kp_ramp_duration", _legacy_ramp_dur))
        self._kp_ramp_duration_steps: int = max(1, int(kp_ramp_dur * policy_hz))
        self._kp_ramp_step: int = 0
        self._kp_ramp_active: bool = False
        self._kp_nominal = np.asarray(cfg_get(real_cfg, "kp_real", [100] * num_actions), dtype=np.float32)
        self._kd_nominal = np.asarray(cfg_get(real_cfg, "kd_real", [2] * num_actions), dtype=np.float32)
        self._kp_ramp_floor_ratio: float = float(cfg_get(cfg, "kp_ramp_floor_ratio", 0.1))

        # Joint safety limits
        self._joint_vel_limit: float = float(
            cfg_get(cfg, "joint_vel_limit", cfg_get(real_cfg, "joint_vel_limit", 10.0))
        )
        joint_pos_lower = cfg_get(real_cfg, "joint_pos_lower", None)
        joint_pos_upper = cfg_get(real_cfg, "joint_pos_upper", None)
        if joint_pos_lower is not None and joint_pos_upper is not None:
            self._joint_pos_lower = np.asarray(joint_pos_lower, dtype=np.float32)
            self._joint_pos_upper = np.asarray(joint_pos_upper, dtype=np.float32)
        else:
            self._joint_pos_lower = None
            self._joint_pos_upper = None

    def compute_kp_ramp_gains(self) -> tuple[Float32Array, Float32Array] | None:
        """Return (kp, kd) for current Kp-ramp step, or None if ramp inactive."""
        if not self._kp_ramp_active:
            return None

        factor = min(1.0, self._kp_ramp_step / self._kp_ramp_duration_steps)
        kp = self._kp_nominal * (self._kp_ramp_floor_ratio + (1.0 - self._kp_ramp_floor_ratio) * factor)

        self._kp_ramp_step += 1
        if self._kp_ramp_step >= self._kp_ramp_duration_steps:
            self._kp_ramp_active = False
            logger.info("Kp ramp complete (%d steps)", self._kp_ramp_duration_steps)

        return np.asarray(kp, dtype=np.float32), self._kd_nominal.copy()

    def start_kp_ramp(self) -> None:
        """Arm the Kp ramp for gradual PD gain increase."""
        self._kp_ramp_step = 0
        self._kp_ramp_active = True
        logger.info(
            "Kp ramp armed: %d steps (%.1fs), floor_ratio=%.2f",
            self._kp_ramp_duration_steps,
            self._kp_ramp_duration_steps / self._policy_hz,
            self._kp_ramp_floor_ratio,
        )

    def send_positions(self, target_dof_pos: Float32Array) -> None:
        """Send position targets, applying Kp ramp gains if active."""
        gains = self.compute_kp_ramp_gains()
        if gains is not None:
            kp, kd = gains
            self._robot.send_positions(target_dof_pos, kp=kp, kd=kd)
        else:
            self._robot.send_positions(target_dof_pos)

    def clip_to_joint_limits(self, target_dof_pos: Float32Array) -> Float32Array:
        """Clip target positions to configured joint limits."""
        if self._joint_pos_lower is not None and self._joint_pos_upper is not None:
            return np.clip(target_dof_pos, self._joint_pos_lower, self._joint_pos_upper)
        return target_dof_pos

    def check_joint_velocity_safety(self) -> bool:
        """Check joint velocities against safety limit. Returns True if violation detected."""
        state = self._robot.get_state()
        max_vel = np.max(np.abs(state.qvel))
        if max_vel > self._joint_vel_limit:
            logger.error(
                "SAFETY: joint velocity %.2f rad/s exceeds limit %.2f",
                max_vel, self._joint_vel_limit,
            )
            return True
        return False
