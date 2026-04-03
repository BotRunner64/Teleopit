"""Sim2Real controller -- state machine + dual-mode control loop.

Supports two operating modes for a physical Unitree G1:
- **Standing**: RL policy maintains balance with fixed default-pose reference
- **Mocap**: RL policy tracks retargeted motion commands

State machine:
    IDLE ──Start──▶ STANDING ──Y──▶ MOCAP ──X──▶ STANDING
    Any  ──L1+R1──▶ DAMPING  ──Start──▶ STANDING
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.controllers.observation import (
    VelCmdObservationBuilder,
    _quat_inv_np,
    _quat_mul_np,
    align_motion_qpos_yaw,
    compute_fixed_yaw_alignment_quat,
    rotate_motion_qpos_by_yaw,
)
from teleopit.controllers.qpos_interpolator import QposInterpolator, QposLowPassFilter
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, RealtimeInputPacket
from teleopit.inputs.udp_bvh_provider import UDPBVHInputProvider
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.common import cfg_get
from teleopit.runtime.factory import build_sim2real_mocap_components
from teleopit.runtime.mocap_session import MocapSessionManager, MocapSessionState
from teleopit.sim.reference_timeline import ReferenceSample, ReferenceTimeline, ReferenceWindow, ReferenceWindowBuilder
from teleopit.sim.realtime_utils import (
    ExponentialVecSmoother,
    RealtimeReferenceManager,
)
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

logger = logging.getLogger(__name__)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


class RobotMode(Enum):
    IDLE = "idle"          # Script waiting, robot controlled by remote
    STANDING = "standing"  # Debug mode, RL policy holds default pose
    MOCAP = "mocap"        # Debug mode, RL policy tracks motion commands
    DAMPING = "damping"    # Emergency stop / recovery


class Sim2RealController:
    """G1 real-robot controller -- standing/mocap dual mode with state machine.

    Standing mode: enter debug mode, RL policy maintains balance at default pose.
    Mocap mode: RL policy tracks retargeted motion commands.
    Both modes share the same RL policy inference pipeline.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mode = RobotMode.IDLE

        self.policy_hz: float = float(cfg_get(cfg, "policy_hz", 50.0))
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # Motion command transition smoothing
        transition_dur = float(cfg_get(cfg, "transition_duration", 0.0) or 0.0)
        self._mocap_transition_duration = transition_dur
        self._pause_resume_transition_duration = float(
            cfg_get(cfg, "pause_resume_transition_duration", transition_dur) or 0.0
        )
        self._qpos_interpolator = QposInterpolator(transition_dur, self.policy_hz)
        self._fixed_ref_yaw_alignment = bool(cfg_get(cfg, "velcmd_fixed_ref_yaw_alignment", True))

        # ---- Real robot (SDK) ----
        real_cfg = cfg_get(cfg, "real_robot")
        self.robot = UnitreeG1Robot(real_cfg)
        self.remote = UnitreeRemote()

        # ---- Mocap pipeline (reuse existing components) ----
        robot_cfg = cfg_get(cfg, "robot")
        mocap_components = build_sim2real_mocap_components(
            cfg,
            self._project_root,
            controller_cls=RLPolicyController,
            obs_builder_cls=VelCmdObservationBuilder,
            pico4_input_cls=Pico4InputProvider,
            udp_bvh_input_cls=UDPBVHInputProvider,
            retargeter_cls=RetargetingModule,
        )
        self.input_provider = mocap_components.input_provider
        self.retargeter = mocap_components.retargeter
        self.policy = mocap_components.controller
        self.obs_builder = mocap_components.obs_builder
        if not bool(getattr(self.policy, "_multi_input", False)):
            raise ValueError(
                "Sim2real requires an ONNX policy with dual inputs ('obs' and 'obs_history')."
            )
        raw_retarget_buffer_enabled = cfg_get(cfg, "retarget_buffer_enabled", True)
        self._retarget_buffer_enabled = bool(raw_retarget_buffer_enabled)
        self._retarget_buffer_window_s = float(cfg_get(cfg, "retarget_buffer_window_s", 0.5))
        if self._retarget_buffer_window_s <= 0.0:
            raise ValueError("retarget_buffer_window_s must be > 0")
        self._reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / self.policy_hz,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        if not self._retarget_buffer_enabled and self._reference_window_builder.requires_timeline:
            raise ValueError(
                "Non-zero reference_steps require retarget_buffer_enabled=true in sim2real so "
                "the realtime reference timeline can sample future/history horizons."
            )
        self._reference_debug_log = bool(cfg_get(cfg, "reference_debug_log", False))
        raw_reference_delay_s = cfg_get(cfg, "retarget_buffer_delay_s", cfg_get(cfg, "realtime_input_delay_s", None))
        provider_fps = float(getattr(self.input_provider, "fps", 30.0))
        self._reference_delay_s = (
            1.0 / max(provider_fps, 1.0)
            if raw_reference_delay_s in (None, "", "null")
            else float(raw_reference_delay_s)
        )
        self._realtime_buffer_low_watermark_steps = self._parse_nonnegative_int(
            cfg_get(cfg, "realtime_buffer_low_watermark_steps", 0),
            field_name="realtime_buffer_low_watermark_steps",
        )
        self._realtime_buffer_high_watermark_steps = self._parse_optional_nonnegative_int(
            cfg_get(cfg, "realtime_buffer_high_watermark_steps", None),
            field_name="realtime_buffer_high_watermark_steps",
        )
        if (
            self._realtime_buffer_high_watermark_steps is not None
            and self._realtime_buffer_high_watermark_steps < self._realtime_buffer_low_watermark_steps
        ):
            raise ValueError(
                "realtime_buffer_high_watermark_steps must be >= realtime_buffer_low_watermark_steps"
            )
        self._realtime_buffer_warmup_steps = self._parse_nonnegative_int(
            cfg_get(cfg, "realtime_buffer_warmup_steps", 0),
            field_name="realtime_buffer_warmup_steps",
        )
        self._pause_resume_warmup_steps = self._parse_nonnegative_int(
            cfg_get(cfg, "pause_resume_warmup_steps", self._realtime_buffer_warmup_steps),
            field_name="pause_resume_warmup_steps",
        )
        self._pause_reset_alignment_on_resume = bool(cfg_get(cfg, "pause_reset_alignment_on_resume", True))
        self._realtime_catchup_enabled = bool(cfg_get(cfg, "realtime_catchup_enabled", False))
        self._realtime_catchup_trigger_steps = self._parse_optional_nonnegative_int(
            cfg_get(cfg, "realtime_catchup_trigger_steps", None),
            field_name="realtime_catchup_trigger_steps",
        )
        self._realtime_catchup_release_steps = self._parse_optional_nonnegative_int(
            cfg_get(cfg, "realtime_catchup_release_steps", None),
            field_name="realtime_catchup_release_steps",
        )
        raw_realtime_catchup_target_delay_s = cfg_get(cfg, "realtime_catchup_target_delay_s", None)
        self._realtime_catchup_target_delay_s = (
            None
            if raw_realtime_catchup_target_delay_s in (None, "", "null")
            else float(raw_realtime_catchup_target_delay_s)
        )
        self._reference_velocity_smoothing_alpha = self._parse_alpha(
            cfg_get(cfg, "reference_velocity_smoothing_alpha", 1.0),
            field_name="reference_velocity_smoothing_alpha",
        )
        self._reference_anchor_velocity_smoothing_alpha = self._parse_alpha(
            cfg_get(cfg, "reference_anchor_velocity_smoothing_alpha", 1.0),
            field_name="reference_anchor_velocity_smoothing_alpha",
        )
        self._reference_qpos_smoothing_alpha = self._parse_alpha(
            cfg_get(cfg, "reference_qpos_smoothing_alpha", 1.0),
            field_name="reference_qpos_smoothing_alpha",
        )
        if self._retarget_buffer_enabled:
            self._reference_window_builder.validate_runtime_support(
                delay_s=self._reference_delay_s,
                window_s=self._retarget_buffer_window_s,
                config_label="Sim2Real reference timeline",
            )
        self._reference_timeline: ReferenceTimeline | None = (
            ReferenceTimeline(window_s=self._retarget_buffer_window_s)
            if self._retarget_buffer_enabled
            else None
        )
        self._reference_manager: RealtimeReferenceManager | None = (
            RealtimeReferenceManager(
                reference_window_builder=self._reference_window_builder,
                low_watermark_steps=self._realtime_buffer_low_watermark_steps,
                high_watermark_steps=self._realtime_buffer_high_watermark_steps,
                warmup_steps=self._realtime_buffer_warmup_steps,
                catchup_enabled=self._realtime_catchup_enabled,
                catchup_trigger_steps=self._realtime_catchup_trigger_steps,
                catchup_release_steps=self._realtime_catchup_release_steps,
                catchup_target_delay_s=self._realtime_catchup_target_delay_s,
            )
            if self._reference_timeline is not None
            else None
        )
        self._motion_joint_vel_smoother = ExponentialVecSmoother(self._reference_velocity_smoothing_alpha)
        self._motion_anchor_lin_vel_smoother = ExponentialVecSmoother(self._reference_anchor_velocity_smoothing_alpha)
        self._motion_anchor_ang_vel_smoother = ExponentialVecSmoother(self._reference_anchor_velocity_smoothing_alpha)
        self._reference_qpos_smoother = QposLowPassFilter(self._reference_qpos_smoothing_alpha)
        self._last_live_packet_seq = -1

        # Default standing pose (29-DOF)
        self.default_angles = np.asarray(
            cfg_get(robot_cfg, "default_angles"), dtype=np.float32
        )
        self.num_actions: int = int(cfg_get(robot_cfg, "num_actions", 29))

        # ---- Standing mode reference qpos ----
        self._standing_qpos = np.zeros(36, dtype=np.float64)
        self._standing_qpos[3] = 1.0  # identity quaternion w=1
        self._standing_qpos[7:36] = self.default_angles.astype(np.float64)

        # ---- Policy state (shared by STANDING and MOCAP) ----
        self._last_action: Float32Array = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None
        self._last_reference_qpos: Float64Array | None = None
        self._mocap_reentry_armed: bool = False
        self._fixed_reference_yaw_quat: Float32Array | None = None
        self._fixed_reference_pivot_pos_w: Float32Array | None = None
        self._mocap_session = MocapSessionManager()
        self._last_commanded_motion_qpos: Float64Array | None = None

        # ---- Kp ramp (gradually increase PD gains after episode-reset) ----
        # Fall back to legacy startup_ramp_duration for backward compatibility.
        _legacy_ramp_dur = cfg_get(cfg, "startup_ramp_duration", cfg_get(real_cfg, "startup_ramp_duration", 2.0))
        kp_ramp_dur = float(cfg_get(cfg, "kp_ramp_duration", _legacy_ramp_dur))
        self._kp_ramp_duration_steps: int = max(1, int(kp_ramp_dur * self.policy_hz))
        self._kp_ramp_step: int = 0
        self._kp_ramp_active: bool = False
        self._kp_nominal = np.asarray(cfg_get(real_cfg, "kp_real", [100] * self.num_actions), dtype=np.float32)
        self._kd_nominal = np.asarray(cfg_get(real_cfg, "kd_real", [2] * self.num_actions), dtype=np.float32)
        self._kp_ramp_floor_ratio: float = float(cfg_get(cfg, "kp_ramp_floor_ratio", 0.1))

        # ---- Joint safety (inspired by GR00T JointSafetyMonitor) ----
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

        # ---- Mocap switch safety ----
        mocap_sw = cfg_get(cfg, "mocap_switch", {})
        self._check_frames: int = int(cfg_get(mocap_sw, "check_frames", 10))
        self._max_pos_value: float = float(cfg_get(mocap_sw, "max_position_value", 5.0))

        logger.info(
            "Sim2RealController ready | mode=IDLE | policy_hz=%.0f",
            self.policy_hz,
        )

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop at policy_hz."""
        logger.info(
            "Control loop started | mode=IDLE | press Start to enter STANDING"
        )
        dt = 1.0 / self.policy_hz

        try:
            while True:
                t0 = time.monotonic()

                # 1. Read remote state
                remote_bytes = self.robot.get_wireless_remote()
                self.remote.update(remote_bytes)

                # 2. Emergency stop (highest priority)
                if self._check_emergency_stop():
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        self._enter_damping()
                    self._sleep_until(t0, dt)
                    continue

                # 3. Mode transitions
                self._handle_transitions()

                # 5. Execute current mode
                if self.mode == RobotMode.STANDING:
                    self._standing_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()

                # 6. Rate control
                self._sleep_until(t0, dt)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt -- shutting down")

    # ------------------------------------------------------------------
    # Mode execution
    # ------------------------------------------------------------------

    def _standing_step(self) -> None:
        """Standing mode: feed fixed default-pose reference to RL policy."""
        robot_state = self.robot.get_state()

        # Build standing reference qpos aligned to robot's current yaw
        qpos = self._standing_qpos.copy()
        align_motion_qpos_yaw(
            np.asarray(robot_state.quat, dtype=np.float32), qpos
        )

        # Standing → zero joint velocity reference
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)

        reference_window = None
        if self._obs_builder_requires_reference_window():
            reference_window = self._build_static_reference_window(qpos)

        obs = self._build_policy_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        target_dof_pos = self._clip_to_joint_limits(target_dof_pos)

        self._send_positions_with_kp_ramp(target_dof_pos)

        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()

    def _mocap_step(self) -> None:
        """Mocap mode: input provider -> retarget -> policy -> update LowCmd targets."""
        if not self.input_provider.is_available():
            logger.warning("Input provider unavailable -- entering damping")
            self._enter_damping()
            return

        try:
            packet = self._fetch_realtime_input_packet()
        except (TimeoutError, RuntimeError):
            logger.warning("Input provider error -- entering damping")
            self._enter_damping()
            return

        self._handle_mocap_control_events(packet.control_events)
        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._paused_mocap_step()
            return

        human_frame = packet.frame
        frame_timestamp = float(packet.timestamp_s)
        frame_seq = int(packet.seq)

        reference_window: ReferenceWindow | None = None
        if self._reference_timeline is not None:
            if int(frame_seq) != self._last_live_packet_seq:
                retargeted = self.retargeter.retarget(human_frame)
                self._reference_timeline.append(
                    self._retarget_to_qpos(retargeted),
                    float(frame_timestamp),
                )
                if self._reference_manager is not None:
                    self._reference_manager.note_realtime_frame()
                self._last_live_packet_seq = int(frame_seq)
            if self._reference_manager is None:
                raise RuntimeError("Realtime reference manager must be initialized when using reference_timeline")
            if not self._reference_manager.warmup_done:
                return
            reference_window, reference_diag = self._reference_manager.sample(
                self._reference_timeline,
                time.monotonic() - self._reference_delay_s,
            )
            if self._reference_debug_log and any(reference_window.fallback_mask()):
                logger.warning(
                    "Reference timeline fallback | buffer_len=%d | steps=%s | modes=%s",
                    len(self._reference_timeline),
                    list(reference_window.reference_steps),
                    list(reference_window.modes()),
                )
            if self._reference_debug_log and reference_diag.used_repeat_padding:
                logger.warning(
                    "Reference timeline repeat padding | buffer_len=%d | future_horizon_steps=%d | steps=%s",
                    len(self._reference_timeline),
                    reference_diag.future_horizon_steps,
                    list(reference_window.reference_steps),
                )
            if self._reference_debug_log and reference_diag.used_catchup:
                logger.warning(
                    "Reference timeline catch-up | requested_base=%.6f | effective_base=%.6f | latest=%.6f | future_horizon_steps=%d",
                    reference_diag.requested_base_time_s,
                    reference_diag.effective_base_time_s,
                    -1.0 if reference_diag.latest_timestamp_s is None else reference_diag.latest_timestamp_s,
                    reference_diag.future_horizon_steps,
                )
            reference_qpos = reference_window.current_sample().qpos
        else:
            retargeted = self.retargeter.retarget(human_frame)
            reference_qpos = self._retarget_to_qpos(retargeted)

        # Robot state from SDK
        robot_state = self.robot.get_state()
        if self._fixed_ref_yaw_alignment:
            reference_qpos = self._align_velcmd_reference_yaw(reference_qpos, robot_state=robot_state)
        raw_reference_qpos = np.asarray(reference_qpos, dtype=np.float64).copy()
        reference_qpos = self._reference_qpos_smoother.apply(reference_qpos)
        qpos = self._qpos_interpolator.apply(reference_qpos)
        if self._mocap_session.state == MocapSessionState.RESUMING and not self._qpos_interpolator.is_active:
            self._mocap_session.finish_resume()
            logger.info("Mocap session -> ACTIVE")

        anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)
        if not self._obs_builder_requires_reference_window():
            if self._qpos_interpolator.is_active:
                true_lin_vel_w, true_ang_vel_w = self._compute_anchor_velocities(reference_qpos)
                blend = np.float32(self._qpos_interpolator.last_alpha)
                raw_anchor_lin_vel_w = np.asarray(true_lin_vel_w * blend, dtype=np.float32)
                raw_anchor_ang_vel_w = np.asarray(true_ang_vel_w * blend, dtype=np.float32)
            else:
                raw_anchor_lin_vel_w, raw_anchor_ang_vel_w = self._compute_anchor_velocities(reference_qpos)
            anchor_lin_vel_w = self._motion_anchor_lin_vel_smoother.apply(raw_anchor_lin_vel_w)
            anchor_ang_vel_w = self._motion_anchor_ang_vel_smoother.apply(raw_anchor_ang_vel_w)

        if qpos.shape[0] < 7 + self.num_actions:
            raise ValueError(
                f"Retargeted qpos too short: {qpos.shape[0]} (need >= {7 + self.num_actions})"
            )
        motion_joint_pos = np.asarray(qpos[7:7 + self.num_actions], dtype=np.float32)
        if self._last_retarget_qpos is None:
            raw_motion_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
        else:
            prev_joint_pos = np.asarray(self._last_retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
            raw_motion_joint_vel = (motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz)
        motion_joint_vel = self._motion_joint_vel_smoother.apply(raw_motion_joint_vel)

        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        obs = self._build_policy_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=anchor_lin_vel_w,
            anchor_ang_vel_w=anchor_ang_vel_w,
            reference_window=reference_window,
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        target_dof_pos = self._clip_to_joint_limits(target_dof_pos)

        self._send_positions_with_kp_ramp(target_dof_pos)

        # Update state
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_reference_qpos = reference_qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()

    def _compute_anchor_velocities(
        self, qpos: Float64Array,
    ) -> tuple[Float32Array, Float32Array]:
        """Compute motion anchor linear/angular velocity in world frame via finite diff."""
        builder = self.obs_builder._base

        cur_pos = np.asarray(qpos[0:3], dtype=np.float32)
        cur_quat = np.asarray(qpos[3:7], dtype=np.float32)
        cur_joints = np.asarray(qpos[7:7 + self.num_actions], dtype=np.float32)
        builder._run_fk(cur_pos, cur_quat, cur_joints)
        cur_anchor_pos = builder._get_body_pos(builder._anchor_body_id).copy()

        if self._last_reference_qpos is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        prev = self._last_reference_qpos
        prev_pos = np.asarray(prev[0:3], dtype=np.float32)
        prev_quat = np.asarray(prev[3:7], dtype=np.float32)
        prev_joints = np.asarray(prev[7:7 + self.num_actions], dtype=np.float32)
        builder._run_fk(prev_pos, prev_quat, prev_joints)
        prev_anchor_pos = builder._get_body_pos(builder._anchor_body_id).copy()

        dt = np.float32(1.0 / self.policy_hz)
        anchor_lin_vel_w = np.asarray(
            (cur_anchor_pos - prev_anchor_pos) / dt, dtype=np.float32,
        )

        # Angular velocity from quaternion difference.
        builder._run_fk(cur_pos, cur_quat, cur_joints)
        cur_anchor_quat = builder._get_body_quat(builder._anchor_body_id).copy()
        builder._run_fk(prev_pos, prev_quat, prev_joints)
        prev_anchor_quat = builder._get_body_quat(builder._anchor_body_id).copy()

        q_delta = _quat_mul_np(cur_anchor_quat, _quat_inv_np(prev_anchor_quat))
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

        return anchor_lin_vel_w, anchor_ang_vel_w

    # ------------------------------------------------------------------
    # Startup ramp and safety
    # ------------------------------------------------------------------

    def _compute_kp_ramp_gains(self) -> tuple[Float32Array, Float32Array] | None:
        """Return (kp, kd) for current Kp-ramp step, or None if ramp inactive.

        Linearly ramps Kp from ``floor_ratio * kp_nominal`` to ``kp_nominal``
        over ``_kp_ramp_duration_steps``.  Kd stays at nominal throughout to
        provide damping from the first step.  Unlike the old position ramp this
        does NOT modify the policy's target position, so the action-state
        causal chain seen by the TemporalCNN stays consistent with training.
        """
        if not self._kp_ramp_active:
            return None

        factor = min(1.0, self._kp_ramp_step / self._kp_ramp_duration_steps)
        kp = self._kp_nominal * (self._kp_ramp_floor_ratio + (1.0 - self._kp_ramp_floor_ratio) * factor)

        self._kp_ramp_step += 1
        if self._kp_ramp_step >= self._kp_ramp_duration_steps:
            self._kp_ramp_active = False
            logger.info("Kp ramp complete (%d steps)", self._kp_ramp_duration_steps)

        return np.asarray(kp, dtype=np.float32), self._kd_nominal.copy()

    def _start_kp_ramp(self) -> None:
        """Arm the Kp ramp for gradual PD gain increase."""
        self._kp_ramp_step = 0
        self._kp_ramp_active = True
        logger.info(
            "Kp ramp armed: %d steps (%.1fs), floor_ratio=%.2f",
            self._kp_ramp_duration_steps,
            self._kp_ramp_duration_steps / self.policy_hz,
            self._kp_ramp_floor_ratio,
        )

    def _send_positions_with_kp_ramp(self, target_dof_pos: Float32Array) -> None:
        """Send position targets, applying Kp ramp gains if active."""
        gains = self._compute_kp_ramp_gains()
        if gains is not None:
            kp, kd = gains
            self.robot.send_positions(target_dof_pos, kp=kp, kd=kd)
        else:
            self.robot.send_positions(target_dof_pos)

    def _clip_to_joint_limits(self, target_dof_pos: Float32Array) -> Float32Array:
        """Clip target positions to configured joint limits."""
        if self._joint_pos_lower is not None and self._joint_pos_upper is not None:
            return np.clip(target_dof_pos, self._joint_pos_lower, self._joint_pos_upper)
        return target_dof_pos

    def _check_joint_velocity_safety(self) -> bool:
        """Check joint velocities against safety limit. Returns True if violation detected."""
        state = self.robot.get_state()
        max_vel = np.max(np.abs(state.qvel))
        if max_vel > self._joint_vel_limit:
            logger.error(
                "SAFETY: joint velocity %.2f rad/s exceeds limit %.2f -- entering damping",
                max_vel, self._joint_vel_limit,
            )
            self._enter_damping()
            return True
        return False

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _check_emergency_stop(self) -> bool:
        """L1 + R1 pressed simultaneously -> emergency stop."""
        return self.remote.LB.pressed and self.remote.RB.pressed

    def _handle_transitions(self) -> None:
        """Handle remote-triggered mode transitions."""
        if self.mode == RobotMode.IDLE:
            if self.remote.start.on_pressed:
                logger.info("Start pressed (from IDLE)")
                self._enter_standing()

        elif self.mode == RobotMode.STANDING:
            reentry_request = self._mocap_reentry_armed and self.remote.Y.pressed
            if self.remote.Y.on_pressed or reentry_request:
                if self._can_switch_to_mocap():
                    if reentry_request and not self.remote.Y.on_pressed:
                        logger.info("Y held after STANDING return -> re-entering MOCAP")
                    else:
                        logger.info("Y pressed -> entering MOCAP")
                    self._transition_to_mocap()
                else:
                    logger.warning("Cannot switch to MOCAP -- input check failed")

        elif self.mode == RobotMode.MOCAP:
            if self.remote.X.on_pressed:
                logger.info("X pressed -> returning to STANDING")
                self._enter_standing()

        elif self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                logger.info("Start pressed (from DAMPING)")
                self._enter_standing()

    # ------------------------------------------------------------------
    # Enter STANDING (from IDLE, MOCAP, or DAMPING)
    # ------------------------------------------------------------------

    def _enter_standing(self) -> None:
        """Enter standing mode: debug mode + RL policy holds default pose.

        Works from IDLE, MOCAP (already in debug mode), or DAMPING.
        """
        prev_mode = self.mode
        already_in_debug = self.mode in (RobotMode.STANDING, RobotMode.MOCAP)

        if not already_in_debug:
            logger.info("Entering debug mode...")
            ok = self.robot.enter_debug_mode()
            if not ok:
                logger.error("Failed to enter debug mode -- staying in %s", self.mode.value)
                return
            time.sleep(0.5)

        # Lock joints to current position (prevent collapse during init)
        logger.info("Locking joints to current position...")
        self.robot.lock_all_joints()
        time.sleep(0.3)

        # Episode-reset semantics: reference = current robot state, full policy reset.
        # This matches training where robot is teleported to reference position at
        # episode start, so policy sees reference ≈ robot state with clean history.
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_reference_qpos = None
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None

        # Always do a full policy reset (episode-reset semantics) to ensure
        # the TemporalCNN history is clean and action-state causality holds.
        self._reset_policy_state()

        # Kp ramp: gradually increase PD gains to avoid torque spike.
        # Unlike the old position ramp, this does NOT break action-state causality.
        self._start_kp_ramp()

        self._mocap_reentry_armed = prev_mode == RobotMode.MOCAP

        self.mode = RobotMode.STANDING
        logger.info("Mode -> STANDING (RL policy maintaining balance at default pose)")

    # ------------------------------------------------------------------
    # STANDING -> MOCAP
    # ------------------------------------------------------------------

    def _can_switch_to_mocap(self) -> bool:
        """Verify input signal is stable and values are reasonable."""
        if not self.input_provider.is_available():
            logger.warning("Mocap check: input provider not available")
            return False

        # For UDP BVH provider, also check if initial data has been received
        if hasattr(self.input_provider, "_frame_ready"):
            if not self.input_provider._frame_ready.is_set():
                logger.warning("Mocap check: no data received yet")
                return False

        # For Pico4 provider, check SDK data availability before calling
        # get_frame() to avoid blocking the control loop for up to
        # pico4_timeout on first frame.
        if hasattr(self.input_provider, "_xrt"):
            if not self.input_provider._xrt.is_body_data_available():
                logger.warning("Mocap check: Pico4 body data not available yet")
                return False

        valid_count = 0
        for _ in range(self._check_frames + 5):
            try:
                frame = self.input_provider.get_frame()
            except (TimeoutError, RuntimeError):
                return False

            all_valid = True
            for bone_name, (pos, quat) in frame.items():
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    all_valid = False
                    break
                if np.any(np.abs(pos) > self._max_pos_value):
                    all_valid = False
                    break
                if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
                    all_valid = False
                    break

            if all_valid:
                valid_count += 1
            else:
                valid_count = 0

            if valid_count >= self._check_frames:
                return True

            time.sleep(0.02)

        logger.warning("Mocap check: only %d/%d valid frames", valid_count, self._check_frames)
        return False

    def _transition_to_mocap(self) -> None:
        """Switch from STANDING -> MOCAP.

        Episode-reset + reference-side interpolation.  The policy state is
        fully reset (clean history, zero last_action) so the TemporalCNN
        starts fresh.  A QposInterpolator smoothly blends the *reference*
        from the current robot state toward incoming live mocap so the policy
        never sees a large instantaneous tracking error.
        """
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_commanded_motion_qpos = init_qpos.copy()
        self._mocap_reentry_armed = False

        # Full episode reset: clean policy state, alignment, timeline.
        self._reset_policy_state()

        # Reference-side interpolation: smoothly blend reference from current
        # robot state toward incoming live mocap.  This is done AFTER the
        # episode reset so the interpolator starts with a clean slate.
        self._arm_qpos_transition(init_qpos, duration_s=self._mocap_transition_duration)

        self.mode = RobotMode.MOCAP
        logger.info("Mode -> MOCAP (tracking motion commands)")

    # ------------------------------------------------------------------
    # Emergency stop / damping
    # ------------------------------------------------------------------

    def _enter_damping(self) -> None:
        """Enter damping mode from any state."""
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP):
            logger.info("DAMPING: sending LowCmd damping...")
            self.robot.set_damping()
            time.sleep(0.5)
            logger.info("DAMPING: exiting debug mode...")
            self.robot.exit_debug_mode()

        self.mode = RobotMode.DAMPING
        self._last_reference_qpos = None
        if self._reference_timeline is not None:
            self._reference_timeline.clear()
        self._last_live_packet_seq = -1
        self._mocap_reentry_armed = False
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        logger.info("Mode -> DAMPING (press Start to re-enter STANDING)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retarget_to_qpos(self, retargeted: object) -> Float64Array:
        """Convert retarget output to 36D qpos (7D root + 29D joints)."""
        if isinstance(retargeted, tuple) and len(retargeted) == 3:
            base_pos = np.asarray(retargeted[0], dtype=np.float64).reshape(-1)
            base_rot = np.asarray(retargeted[1], dtype=np.float64).reshape(-1)
            joint_pos = np.asarray(retargeted[2], dtype=np.float64).reshape(-1)
            qpos = np.concatenate((base_pos, base_rot, joint_pos))
        else:
            qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < 36:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= 36)")
        return qpos

    def _align_velcmd_reference_yaw(
        self,
        qpos: Float64Array,
        robot_state: object | None,
    ) -> Float64Array:
        aligned_qpos = qpos.copy()

        if self._fixed_reference_yaw_quat is None:
            if robot_state is None:
                robot_state = self.robot.get_state()
            robot_quat = np.asarray(getattr(robot_state, "quat"), dtype=np.float32)
            self._fixed_reference_yaw_quat = compute_fixed_yaw_alignment_quat(
                robot_quat,
                np.asarray(aligned_qpos[3:7], dtype=np.float32),
            )
            self._fixed_reference_pivot_pos_w = np.asarray(aligned_qpos[0:3], dtype=np.float32).copy()

        rotate_motion_qpos_by_yaw(
            aligned_qpos,
            self._fixed_reference_yaw_quat,
            self._fixed_reference_pivot_pos_w,
        )
        return aligned_qpos

    def _build_static_reference_window(self, qpos: Float64Array) -> ReferenceWindow:
        base_time_s = time.monotonic()
        reference_steps = tuple(self._reference_window_builder.reference_steps)
        qpos_copy = np.asarray(qpos, dtype=np.float64).reshape(-1).copy()
        samples = tuple(
            ReferenceSample(
                qpos=qpos_copy.copy(),
                timestamp_s=base_time_s + float(step) / self.policy_hz,
                mode="static_reference",
                used_fallback=False,
                older_timestamp_s=base_time_s + float(step) / self.policy_hz,
                newer_timestamp_s=base_time_s + float(step) / self.policy_hz,
                alpha=None,
            )
            for step in reference_steps
        )
        return ReferenceWindow(
            base_time_s=base_time_s,
            policy_dt_s=1.0 / self.policy_hz,
            reference_steps=reference_steps,
            samples=samples,
        )

    def _obs_builder_requires_reference_window(self) -> bool:
        return bool(getattr(self.obs_builder, "requires_reference_window", False)) or callable(
            getattr(self.obs_builder, "build_with_reference_window", None)
        )

    def _align_reference_window(
        self,
        reference_window: ReferenceWindow | None,
        robot_state: object,
    ) -> ReferenceWindow | None:
        if reference_window is None or not self._fixed_ref_yaw_alignment:
            return reference_window

        current_qpos = np.asarray(reference_window.current_sample().qpos, dtype=np.float64).reshape(-1)
        if self._fixed_reference_yaw_quat is None:
            robot_quat = np.asarray(getattr(robot_state, "quat"), dtype=np.float32)
            self._fixed_reference_yaw_quat = compute_fixed_yaw_alignment_quat(
                robot_quat,
                np.asarray(current_qpos[3:7], dtype=np.float32),
            )
            self._fixed_reference_pivot_pos_w = np.asarray(current_qpos[0:3], dtype=np.float32).copy()

        aligned_samples: list[ReferenceSample] = []
        for sample in reference_window.samples:
            aligned_qpos = np.asarray(sample.qpos, dtype=np.float64).reshape(-1).copy()
            rotate_motion_qpos_by_yaw(
                aligned_qpos,
                self._fixed_reference_yaw_quat,
                self._fixed_reference_pivot_pos_w,
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

        return ReferenceWindow(
            base_time_s=float(reference_window.base_time_s),
            policy_dt_s=float(reference_window.policy_dt_s),
            reference_steps=tuple(reference_window.reference_steps),
            samples=tuple(aligned_samples),
        )

    def _build_policy_observation(
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
        build_with_reference_window = getattr(self.obs_builder, "build_with_reference_window", None)
        if callable(build_with_reference_window):
            aligned_reference_window = self._align_reference_window(reference_window, robot_state)
            obs = build_with_reference_window(
                robot_state,
                aligned_reference_window,
                motion_qpos,
                last_action,
            )
        else:
            obs = self.obs_builder.build(
                robot_state,
                motion_qpos,
                motion_joint_vel,
                last_action,
                anchor_lin_vel_w,
                anchor_ang_vel_w,
            )
        return np.asarray(obs, dtype=np.float32)

    def _validate_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        """Fail-fast validation for policy input observation dimension."""
        expected = getattr(self.policy, "_expected_obs_dim", None)
        if not isinstance(expected, int) or expected <= 0:
            return obs
        if obs.shape[0] != expected:
            raise ValueError(
                f"Observation dimension mismatch: obs_builder produced {obs.shape[0]}, "
                f"but policy expects {expected}. "
                "Use a matching mjlab-aligned ONNX policy; automatic pad/trim is disabled."
            )
        return obs

    def _reset_policy_state(self) -> None:
        """Full episode-reset: clear all policy state so the TemporalCNN sees
        a clean start identical to training episode reset."""
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._qpos_interpolator.reset()
        self._reset_mocap_reference_state()
        self._reset_reference_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self.obs_builder.reset()

    def _reset_mocap_reference_state(self, *, warmup_steps: int | None = None) -> None:
        """Reset mocap-specific reference state without disrupting policy observation continuity.

        Unlike ``_reset_policy_state``, this preserves ``_last_action``, the
        policy history buffer, and the observation builder state so that the
        TemporalCNN sees a continuous observation stream across mode switches.
        """
        if self._reference_timeline is not None:
            self._reference_timeline.clear()
        if self._reference_manager is not None:
            self._reference_manager.set_warmup_steps(
                self._realtime_buffer_warmup_steps if warmup_steps is None else warmup_steps
            )
            self._reference_manager.reset()
        self._motion_joint_vel_smoother.reset()
        self._motion_anchor_lin_vel_smoother.reset()
        self._motion_anchor_ang_vel_smoother.reset()
        self._reference_qpos_smoother.reset()
        self._last_reference_qpos = None
        self._last_live_packet_seq = -1

    def _reset_reference_alignment(self) -> None:
        self._fixed_reference_yaw_quat = None
        self._fixed_reference_pivot_pos_w = None

    def _arm_qpos_transition(self, start_qpos: Float64Array, *, duration_s: float) -> None:
        self._qpos_interpolator.reset()
        self._qpos_interpolator.configure(duration_s)
        self._qpos_interpolator.start(start_qpos)

    def _fetch_realtime_input_packet(self) -> RealtimeInputPacket[object]:
        get_realtime_input_packet = getattr(self.input_provider, "get_realtime_input_packet", None)
        if callable(get_realtime_input_packet):
            return get_realtime_input_packet()

        get_packet = getattr(self.input_provider, "get_frame_packet", None)
        if callable(get_packet):
            frame, frame_timestamp, frame_seq = get_packet()
            return RealtimeInputPacket(
                frame=frame,
                timestamp_s=float(frame_timestamp),
                seq=int(frame_seq),
                control_events=(),
            )

        frame = self.input_provider.get_frame()
        return RealtimeInputPacket(
            frame=frame,
            timestamp_s=time.monotonic(),
            seq=self._last_live_packet_seq + 1,
            control_events=(),
        )

    def _handle_mocap_control_events(self, control_events: tuple[ControlEvent, ...]) -> None:
        for event in control_events:
            if event.event_type != ControlEventType.TOGGLE_PAUSE:
                continue
            if self._mocap_session.state == MocapSessionState.PAUSED:
                self._resume_paused_mocap()
            else:
                self._pause_active_mocap()

    def _pause_active_mocap(self) -> None:
        # Episode-reset semantics: treat pause as a new episode starting at
        # the hold pose.  Full policy reset ensures TemporalCNN history is
        # clean -- no stale frames from the previous motion that would create
        # an OOD discontinuity (reference jumps from motion to static).
        hold_qpos = self._resolve_mocap_hold_qpos()
        self._last_retarget_qpos = hold_qpos.copy()
        self._last_reference_qpos = hold_qpos.copy()
        self._last_commanded_motion_qpos = hold_qpos.copy()

        # Reset policy state (clears last_action, history, smoothers, etc.)
        # Note: _reset_policy_state resets _mocap_session to ACTIVE, so we
        # must call pause() *after* it to set the correct PAUSED state.
        self._reset_policy_state()
        self._mocap_session.pause(hold_qpos)
        logger.info("Mocap session -> PAUSED (episode-reset)")

    def _resume_paused_mocap(self) -> None:
        # Episode-reset + reference-side interpolation.
        #
        # 1. Reference starts at current robot state (not hold_qpos) so there
        #    is no reference-state mismatch at the moment of resume.
        # 2. Full policy reset ensures clean TemporalCNN history.
        # 3. QposInterpolator smoothly blends reference from robot state
        #    toward incoming live mocap so the policy never sees a large
        #    instantaneous tracking error.

        state = self.robot.get_state()
        resume_qpos = np.zeros(36, dtype=np.float64)
        resume_qpos[3:7] = state.quat.astype(np.float64)
        resume_qpos[7:36] = state.qpos.astype(np.float64)

        self._last_retarget_qpos = resume_qpos.copy()
        self._last_commanded_motion_qpos = resume_qpos.copy()

        # Full policy reset -- clean history, zero last_action, smoothers,
        # timeline, alignment.  Also resets _mocap_session to ACTIVE.
        self._reset_policy_state()

        # Override warmup steps for the resume-specific buffer warmup.
        if self._reference_manager is not None:
            self._reference_manager.set_warmup_steps(self._pause_resume_warmup_steps)
            self._reference_manager.reset()

        # Reference-side interpolation: smoothly blend from current robot
        # state toward incoming live mocap.
        self._arm_qpos_transition(resume_qpos, duration_s=self._pause_resume_transition_duration)

        logger.info("Mocap session -> ACTIVE (episode-reset + reference interpolation)")

    def _resolve_mocap_hold_qpos(self) -> Float64Array:
        if self._last_commanded_motion_qpos is not None:
            return self._last_commanded_motion_qpos.copy()
        if self._last_retarget_qpos is not None:
            return np.asarray(self._last_retarget_qpos, dtype=np.float64).copy()
        state = self.robot.get_state()
        hold_qpos = np.zeros(36, dtype=np.float64)
        hold_qpos[3:7] = np.asarray(state.quat, dtype=np.float64)
        hold_qpos[7:36] = np.asarray(state.qpos, dtype=np.float64)
        return hold_qpos

    def _paused_mocap_step(self) -> None:
        hold_qpos = self._mocap_session.hold_qpos
        if hold_qpos is None:
            raise RuntimeError("Paused mocap session is missing a hold_qpos")

        robot_state = self.robot.get_state()
        qpos = np.asarray(hold_qpos, dtype=np.float64).copy()
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        reference_window = None
        if self._obs_builder_requires_reference_window():
            reference_window = self._build_static_reference_window(qpos)

        obs = self._build_policy_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        target_dof_pos = self._clip_to_joint_limits(target_dof_pos)

        self._send_positions_with_kp_ramp(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_reference_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()

    @staticmethod
    def _parse_nonnegative_int(value: object, *, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a non-negative integer, got {value}")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")
        return parsed

    @staticmethod
    def _parse_alpha(value: object, *, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be in (0, 1], got {value}")
        parsed = float(value)
        if parsed <= 0.0 or parsed > 1.0:
            raise ValueError(f"{field_name} must be in (0, 1], got {value}")
        return parsed

    @staticmethod
    def _parse_optional_nonnegative_int(value: object | None, *, field_name: str) -> int | None:
        if value in (None, "", "null"):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a non-negative integer, got {value}")
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{field_name} must be >= 0, got {value}")
        return parsed

    @staticmethod
    def _sleep_until(t0: float, dt: float) -> None:
        """Sleep to maintain control frequency."""
        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean shutdown."""
        logger.info("Shutting down Sim2RealController")
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP):
            try:
                self.robot.set_damping()
                time.sleep(0.5)
            except Exception:
                pass
            try:
                self.robot.exit_debug_mode()
            except Exception:
                pass
        try:
            self.input_provider.close()
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass
