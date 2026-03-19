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
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.udp_bvh_provider import UDPBVHInputProvider
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.common import cfg_get
from teleopit.runtime.factory import build_sim2real_mocap_components
from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindowBuilder
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
        provider_fps = float(getattr(self.input_provider, "fps", self.policy_hz))
        self._reference_delay_s = (
            1.0 / provider_fps
            if raw_reference_delay_s in (None, "", "null")
            else float(raw_reference_delay_s)
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
                ls = self.robot.get_lowstate()
                if ls is not None:
                    remote_bytes = bytes(ls.wireless_remote)
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

                # 4. Execute current mode
                if self.mode == RobotMode.STANDING:
                    self._standing_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()

                # 5. Rate control
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

        obs = self.obs_builder.build(
            robot_state, motion_qpos, motion_joint_vel, self._last_action,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        self.robot.send_positions(target_dof_pos)

        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()

    def _mocap_step(self) -> None:
        """Mocap mode: input provider -> retarget -> policy -> update LowCmd targets."""
        if not self.input_provider.is_available():
            logger.warning("Input provider unavailable -- entering damping")
            self._enter_damping()
            return

        try:
            get_packet = getattr(self.input_provider, "get_frame_packet", None)
            if callable(get_packet):
                human_frame, frame_timestamp, frame_seq = get_packet()
            else:
                human_frame = self.input_provider.get_frame()
                frame_timestamp = time.monotonic()
                frame_seq = self._last_live_packet_seq + 1
        except (TimeoutError, RuntimeError):
            logger.warning("Input provider error -- entering damping")
            self._enter_damping()
            return

        if self._reference_timeline is not None:
            if int(frame_seq) != self._last_live_packet_seq:
                retargeted = self.retargeter.retarget(human_frame)
                self._reference_timeline.append(
                    self._retarget_to_qpos(retargeted),
                    float(frame_timestamp),
                )
                self._last_live_packet_seq = int(frame_seq)
            reference_window = self._reference_window_builder.sample(
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
            reference_qpos = reference_window.current_sample().qpos
        else:
            retargeted = self.retargeter.retarget(human_frame)
            reference_qpos = self._retarget_to_qpos(retargeted)

        # Robot state from SDK
        robot_state = self.robot.get_state()
        if self._fixed_ref_yaw_alignment:
            reference_qpos = self._align_velcmd_reference_yaw(reference_qpos, robot_state=robot_state)
        qpos = self._qpos_interpolator.apply(reference_qpos)

        if self._qpos_interpolator.is_active:
            true_lin_vel_w, true_ang_vel_w = self._compute_anchor_velocities(reference_qpos)
            blend = np.float32(self._qpos_interpolator.last_alpha)
            anchor_lin_vel_w = np.asarray(true_lin_vel_w * blend, dtype=np.float32)
            anchor_ang_vel_w = np.asarray(true_ang_vel_w * blend, dtype=np.float32)
        else:
            anchor_lin_vel_w, anchor_ang_vel_w = self._compute_anchor_velocities(reference_qpos)

        if qpos.shape[0] < 7 + self.num_actions:
            raise ValueError(
                f"Retargeted qpos too short: {qpos.shape[0]} (need >= {7 + self.num_actions})"
            )
        motion_joint_pos = np.asarray(qpos[7:7 + self.num_actions], dtype=np.float32)
        if self._last_retarget_qpos is None:
            motion_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
        else:
            prev_joint_pos = np.asarray(self._last_retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
            motion_joint_vel = (motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz)

        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        obs = self.obs_builder.build(
            robot_state, motion_qpos, motion_joint_vel, self._last_action,
            anchor_lin_vel_w, anchor_ang_vel_w,
        )
        obs = self._validate_observation_for_policy(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)

        # Update position targets (250Hz thread handles publishing)
        self.robot.send_positions(target_dof_pos)

        # Update state
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_reference_qpos = reference_qpos.copy()

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

        # Initialize policy state
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_reference_qpos = None
        self._reset_policy_state()
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

        Already in debug mode from STANDING, so just read current state
        and start interpolation for smooth transition.
        """
        # Read current state as initial reference for interpolation
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_reference_qpos = None
        self._fixed_reference_yaw_quat = None
        self._fixed_reference_pivot_pos_w = None
        self._qpos_interpolator.reset()
        self._qpos_interpolator.start(init_qpos)
        self._reset_policy_state()
        self._mocap_reentry_armed = False

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
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        if self._reference_timeline is not None:
            self._reference_timeline.clear()
        self._last_live_packet_seq = -1
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        self.obs_builder.reset()

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
