"""Reference processing for Sim2Real controller.

Encapsulates retargeting conversion, yaw alignment, anchor velocity computation,
observation building, frame validation, and smoothing.
Delegates core algorithms to teleopit.controllers.reference_processing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.controllers import reference_processing as ref_proc
from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.controllers.qpos_interpolator import QposLowPassFilter
from teleopit.sim.realtime_utils import ExponentialVecSmoother
from teleopit.sim.reference_timeline import ReferenceWindow

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
logger = logging.getLogger(__name__)


class Sim2RealReferenceProcessor:
    """Retargeting, yaw alignment, anchor velocities, observation building, and smoothing."""

    def __init__(
        self,
        obs_builder: VelCmdObservationBuilder,
        policy: Any,
        policy_hz: float,
        num_actions: int,
        reference_velocity_smoothing_alpha: float,
        reference_anchor_velocity_smoothing_alpha: float,
        reference_qpos_smoothing_alpha: float,
        max_pos_value: float,
    ) -> None:
        self._obs_builder = obs_builder
        self._policy = policy
        self._policy_hz = policy_hz
        self._num_actions = num_actions
        self._max_pos_value = max_pos_value

        # Yaw alignment state (lazy-init)
        self._fixed_reference_yaw_quat: Float32Array | None = None
        self._fixed_reference_pivot_pos_w: Float32Array | None = None
        self._fixed_reference_xy_offset_w: Float32Array | None = None
        self._reference_alignment_target_xy_w: Float32Array | None = None

        # Smoothers
        self._motion_joint_vel_smoother = ExponentialVecSmoother(reference_velocity_smoothing_alpha)
        self._motion_anchor_lin_vel_smoother = ExponentialVecSmoother(reference_anchor_velocity_smoothing_alpha)
        self._motion_anchor_ang_vel_smoother = ExponentialVecSmoother(reference_anchor_velocity_smoothing_alpha)
        self._reference_qpos_smoother = QposLowPassFilter(reference_qpos_smoothing_alpha)

        # Last reference qpos for velocity computation
        self._last_reference_qpos: Float64Array | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def last_reference_qpos(self) -> Float64Array | None:
        return self._last_reference_qpos

    @last_reference_qpos.setter
    def last_reference_qpos(self, value: Float64Array | None) -> None:
        self._last_reference_qpos = value

    # ------------------------------------------------------------------
    # Retargeting and validation
    # ------------------------------------------------------------------

    def retarget_to_qpos(self, retargeted: object) -> Float64Array:
        """Convert retarget output to 36D qpos (7D root + 29D joints)."""
        return ref_proc.retarget_to_qpos(retargeted)

    def frame_is_valid(self, frame: dict[str, tuple[np.ndarray, np.ndarray]]) -> bool:
        for pos, quat in frame.values():
            if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                return False
            if np.any(np.abs(pos) > self._max_pos_value):
                return False
            if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
                return False
        return True

    # ------------------------------------------------------------------
    # Yaw alignment
    # ------------------------------------------------------------------

    def align_reference_yaw(
        self,
        qpos: Float64Array,
        robot_state: object | None = None,
    ) -> Float64Array:
        if robot_state is None:
            raise ValueError("robot_state required for yaw alignment")
        robot_quat = np.asarray(getattr(robot_state, "quat"), dtype=np.float32)
        (
            aligned,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
        ) = ref_proc.align_reference_yaw(
            qpos,
            robot_quat,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
            self._reference_alignment_target_xy_w,
        )
        return aligned

    def align_reference_window(
        self,
        reference_window: ReferenceWindow | None,
        robot_state: object,
    ) -> ReferenceWindow | None:
        if reference_window is None:
            return reference_window
        robot_quat = np.asarray(getattr(robot_state, "quat"), dtype=np.float32)
        (
            aligned,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
        ) = ref_proc.align_reference_window(
            reference_window,
            robot_quat,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
            self._fixed_reference_xy_offset_w,
            self._reference_alignment_target_xy_w,
        )
        return aligned

    def reset_alignment(self, target_qpos: Float64Array | None = None) -> None:
        self._fixed_reference_yaw_quat = None
        self._fixed_reference_pivot_pos_w = None
        self._fixed_reference_xy_offset_w = None
        self._reference_alignment_target_xy_w = (
            None
            if target_qpos is None
            else np.asarray(target_qpos[0:2], dtype=np.float32).reshape(2).copy()
        )

    # ------------------------------------------------------------------
    # Anchor velocity computation
    # ------------------------------------------------------------------

    def compute_anchor_velocities(
        self, qpos: Float64Array,
    ) -> tuple[Float32Array, Float32Array]:
        """Compute motion anchor linear/angular velocity in world frame via finite diff."""
        return ref_proc.compute_anchor_velocities(
            self._obs_builder._base, qpos, self._last_reference_qpos,
            self._num_actions, self._policy_hz,
        )

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(
        self,
        *,
        robot_state: object,
        motion_qpos: Float32Array,
        motion_joint_vel: Float32Array,
        last_action: Float32Array,
        anchor_lin_vel_w: Float32Array,
        anchor_ang_vel_w: Float32Array,
        reference_window: ReferenceWindow | None,
    ) -> Float32Array:
        aligned_reference_window = self.align_reference_window(reference_window, robot_state)
        return ref_proc.dispatch_build_observation(
            self._obs_builder, robot_state, reference_window, aligned_reference_window,
            motion_qpos, motion_joint_vel, last_action,
            anchor_lin_vel_w, anchor_ang_vel_w,
        )

    def validate_observation(self, obs: Float32Array) -> Float32Array:
        """Fail-fast validation for policy input observation dimension."""
        expected = getattr(self._policy, "_expected_obs_dim", None)
        if not isinstance(expected, int) or expected <= 0:
            return obs
        if obs.shape[0] != expected:
            raise ValueError(
                f"Observation dimension mismatch: obs_builder produced {obs.shape[0]}, "
                f"but policy expects {expected}. "
                "Use a matching mjlab-aligned ONNX policy; automatic pad/trim is disabled."
            )
        return obs

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def apply_qpos_smoothing(self, qpos: Float64Array) -> Float64Array:
        return self._reference_qpos_smoother.apply(qpos)

    def apply_joint_vel_smoothing(self, vel: Float32Array) -> Float32Array:
        return self._motion_joint_vel_smoother.apply(vel)

    def apply_anchor_vel_smoothing(
        self, lin: Float32Array, ang: Float32Array,
    ) -> tuple[Float32Array, Float32Array]:
        return (
            self._motion_anchor_lin_vel_smoother.apply(lin),
            self._motion_anchor_ang_vel_smoother.apply(ang),
        )

    def reset_smoothers(self) -> None:
        self._motion_joint_vel_smoother.reset()
        self._motion_anchor_lin_vel_smoother.reset()
        self._motion_anchor_ang_vel_smoother.reset()
        self._reference_qpos_smoother.reset()
        self._last_reference_qpos = None
