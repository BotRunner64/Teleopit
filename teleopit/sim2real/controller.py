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

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS, ROOT_DIM
from teleopit.controllers.observation import (
    VelCmdObservationBuilder,
    align_motion_qpos_yaw,
)
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.bvh_provider import BVHInputProvider
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.pico_video import PicoVideoRuntime, parse_pico_video_config
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, RealtimeInputPacket
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.common import cfg_get
from teleopit.runtime.reference_config import parse_reference_config
from teleopit.runtime.factory import build_sim2real_mocap_components
from teleopit.runtime.mocap_session import MocapSessionManager, MocapSessionState
from teleopit.runtime.offline_playback import OfflinePlaybackController
from teleopit.sim.reference_motion import OfflineReferenceMotion
from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindow, ReferenceWindowBuilder
from teleopit.sim.reference_utils import (
    build_offline_reference_window,
    build_static_reference_window,
    obs_builder_requires_reference_window,
)
from teleopit.sim.realtime_utils import RealtimeReferenceManager
from teleopit.sim2real.dexterous_hand import build_linkerhand_runtime
from teleopit.sim2real.reference_processor import Sim2RealReferenceProcessor
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.safety import Sim2RealSafetyManager
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

        self._init_components(cfg)
        self._init_reference_config(cfg)
        self._safety = Sim2RealSafetyManager(cfg, self.robot, self.policy_hz, self.num_actions)

        logger.info(
            "Sim2RealController ready | mode=IDLE | policy_hz=%.0f",
            self.policy_hz,
        )

    def _init_components(self, cfg: Any) -> None:
        """Build robot hardware and mocap pipeline components."""
        real_cfg = cfg_get(cfg, "real_robot")
        self.robot = UnitreeG1Robot(real_cfg)
        self.remote = UnitreeRemote()

        robot_cfg = cfg_get(cfg, "robot")
        mocap_components = build_sim2real_mocap_components(
            cfg,
            self._project_root,
            controller_cls=RLPolicyController,
            obs_builder_cls=VelCmdObservationBuilder,
            bvh_input_cls=BVHInputProvider,
            pico4_input_cls=Pico4InputProvider,
            retargeter_cls=RetargetingModule,
        )
        self.input_provider = mocap_components.input_provider
        self.retargeter = mocap_components.retargeter
        self.policy = mocap_components.controller
        self.obs_builder = mocap_components.obs_builder
        self._video_runtime = PicoVideoRuntime(
            provider=self.input_provider,
            config=parse_pico_video_config(cfg_get(cfg, "input", {})),
            mode="sim2real",
        )
        self._hand_runtime = build_linkerhand_runtime(cfg, self.input_provider)
        self._offline_reference: OfflineReferenceMotion | None = None
        self._offline_playback: OfflinePlaybackController | None = None
        if hasattr(self.input_provider, "__len__") and hasattr(self.input_provider, "get_frame_by_index"):
            playback_cfg = cfg_get(cfg, "playback", {})
            self._offline_reference = OfflineReferenceMotion(self.input_provider, self.retargeter)
            self._offline_playback = OfflinePlaybackController(
                duration_s=self._offline_reference.duration_s,
                step_dt_s=1.0 / self.policy_hz,
                pause_on_end=bool(cfg_get(playback_cfg, "pause_on_end", True)),
            )
        if not bool(getattr(self.policy, "_multi_input", False)):
            raise ValueError(
                "Sim2real requires an ONNX policy with dual inputs ('obs' and 'obs_history')."
            )

        # Default standing pose (29-DOF)
        self.default_angles = np.asarray(
            cfg_get(robot_cfg, "default_angles"), dtype=np.float32
        )
        self.num_actions: int = int(cfg_get(robot_cfg, "num_actions", NUM_JOINTS))

        # Standing mode reference qpos
        self._standing_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        self._standing_qpos[3] = 1.0  # identity quaternion w=1
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)

        # Policy state (shared by STANDING and MOCAP)
        self._last_action: Float32Array = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None
        self._mocap_reentry_armed: bool = False
        self._mocap_session = MocapSessionManager()
        self._last_commanded_motion_qpos: Float64Array | None = None

    def _init_reference_config(self, cfg: Any) -> None:
        """Parse reference-window / realtime-buffer configuration."""
        provider_fps = float(getattr(self.input_provider, "fps", 30.0))
        self._ref_cfg = parse_reference_config(cfg, provider_fps=provider_fps)
        rc = self._ref_cfg

        self._reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / self.policy_hz,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        if not rc.retarget_buffer_enabled and self._reference_window_builder.requires_timeline:
            raise ValueError(
                "Non-zero reference_steps require retarget_buffer_enabled=true in sim2real so "
                "the realtime reference timeline can sample future/history horizons."
            )
        if rc.retarget_buffer_enabled:
            self._reference_window_builder.validate_runtime_support(
                delay_s=rc.reference_delay_s,
                window_s=rc.retarget_buffer_window_s,
                config_label="Sim2Real reference timeline",
            )
        self._reference_timeline: ReferenceTimeline | None = (
            ReferenceTimeline(window_s=rc.retarget_buffer_window_s)
            if rc.retarget_buffer_enabled
            else None
        )
        self._reference_manager: RealtimeReferenceManager | None = (
            RealtimeReferenceManager(
                reference_window_builder=self._reference_window_builder,
                low_watermark_steps=rc.realtime_buffer_low_watermark_steps,
                high_watermark_steps=rc.realtime_buffer_high_watermark_steps,
                warmup_steps=rc.realtime_buffer_warmup_steps,
                catchup_enabled=rc.realtime_catchup_enabled,
                catchup_trigger_steps=rc.realtime_catchup_trigger_steps,
                catchup_release_steps=rc.realtime_catchup_release_steps,
                catchup_target_delay_s=rc.realtime_catchup_target_delay_s,
            )
            if self._reference_timeline is not None
            else None
        )
        mocap_sw = cfg_get(cfg, "mocap_switch", {})
        self._ref_proc = Sim2RealReferenceProcessor(
            obs_builder=self.obs_builder,
            policy=self.policy,
            policy_hz=self.policy_hz,
            num_actions=self.num_actions,
            reference_velocity_smoothing_alpha=rc.reference_velocity_smoothing_alpha,
            reference_anchor_velocity_smoothing_alpha=rc.reference_anchor_velocity_smoothing_alpha,
            reference_qpos_smoothing_alpha=rc.reference_qpos_smoothing_alpha,
            max_pos_value=float(cfg_get(mocap_sw, "max_position_value", 5.0)),
        )
        self._last_live_packet_seq = -1

        # Mocap switch safety
        mocap_sw = cfg_get(cfg, "mocap_switch", {})
        self._check_frames: int = int(cfg_get(mocap_sw, "check_frames", 10))

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
            self._video_runtime.start()
            self._hand_runtime.start()
            while True:
                t0 = time.monotonic()
                self._video_runtime.tick()

                # 1. Read remote state
                remote_bytes = self.robot.get_wireless_remote()
                self.remote.update(remote_bytes)

                # 2. Emergency stop (highest priority)
                if self.remote.LB.pressed and self.remote.RB.pressed:
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        self._enter_damping()
                    self._tick_dexterous_hand()
                    self._sleep_until(t0, dt)
                    continue

                # 3. Mode transitions
                self._handle_transitions()

                # 5. Execute current mode
                if self.mode == RobotMode.STANDING:
                    self._standing_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()

                self._tick_dexterous_hand()

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
        if obs_builder_requires_reference_window(self.obs_builder):
            reference_window = build_static_reference_window(qpos, self._reference_window_builder, self.policy_hz)

        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self._ref_proc.validate_observation(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        target_dof_pos = self._safety.clip_to_joint_limits(target_dof_pos)

        self._safety.send_positions(target_dof_pos)

        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()

    def _mocap_step(self) -> None:
        """Mocap mode: input provider -> retarget -> policy -> update LowCmd targets."""
        if self._offline_reference is not None:
            self._offline_mocap_step()
            return

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
                    self._ref_proc.retarget_to_qpos(retargeted),
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
                time.monotonic() - self._ref_cfg.reference_delay_s,
            )
            if self._ref_cfg.reference_debug_log and any(reference_window.fallback_mask()):
                logger.warning(
                    "Reference timeline fallback | buffer_len=%d | steps=%s | modes=%s",
                    len(self._reference_timeline),
                    list(reference_window.reference_steps),
                    list(reference_window.modes()),
                )
            if self._ref_cfg.reference_debug_log and reference_diag.used_repeat_padding:
                logger.warning(
                    "Reference timeline repeat padding | buffer_len=%d | future_horizon_steps=%d | steps=%s",
                    len(self._reference_timeline),
                    reference_diag.future_horizon_steps,
                    list(reference_window.reference_steps),
                )
            if self._ref_cfg.reference_debug_log and reference_diag.used_catchup:
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
            reference_qpos = self._ref_proc.retarget_to_qpos(retargeted)

        robot_state = self.robot.get_state()
        self._execute_mocap_pipeline(reference_qpos, robot_state, reference_window)

    def _offline_mocap_step(self) -> None:
        if self._offline_reference is None or self._offline_playback is None:
            raise RuntimeError("Offline playback step requires an offline reference motion")

        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._paused_mocap_step()
            return

        sample_time_s = self._offline_playback.current_time_s
        sampled = self._offline_reference.sample(sample_time_s)
        if sampled is None:
            self._hold_completed_offline_playback(self._resolve_mocap_hold_qpos())
            self._paused_mocap_step()
            return

        reference_window: ReferenceWindow | None = None
        if obs_builder_requires_reference_window(self.obs_builder):
            reference_window = build_offline_reference_window(
                self._offline_reference,
                sample_time_s,
                self._reference_window_builder,
                self.policy_hz,
            )

        reference_qpos = np.asarray(sampled.qpos, dtype=np.float64).copy()
        robot_state = self.robot.get_state()
        self._execute_mocap_pipeline(reference_qpos, robot_state, reference_window)

        if self._offline_playback.advance():
            self._hold_completed_offline_playback(self._last_commanded_motion_qpos)

    def _execute_mocap_pipeline(
        self,
        reference_qpos: Float64Array,
        robot_state: object,
        reference_window: ReferenceWindow | None,
    ) -> None:
        """Shared mocap control pipeline: align → smooth → interpolate → infer → send."""
        reference_qpos = self._ref_proc.align_reference_yaw(reference_qpos, robot_state=robot_state)
        reference_qpos = self._ref_proc.apply_qpos_smoothing(reference_qpos)
        qpos = self._qpos_interpolator.apply(reference_qpos)

        # Compute joint velocities via finite difference
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
        motion_joint_vel = self._ref_proc.apply_joint_vel_smoothing(raw_motion_joint_vel)

        # Compute anchor velocities (with interpolator blending if active)
        anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)
        if not obs_builder_requires_reference_window(self.obs_builder):
            if self._qpos_interpolator.is_active:
                true_lin_vel_w, true_ang_vel_w = self._ref_proc.compute_anchor_velocities(reference_qpos)
                blend = np.float32(self._qpos_interpolator.last_alpha)
                raw_anchor_lin_vel_w = np.asarray(true_lin_vel_w * blend, dtype=np.float32)
                raw_anchor_ang_vel_w = np.asarray(true_ang_vel_w * blend, dtype=np.float32)
            else:
                raw_anchor_lin_vel_w, raw_anchor_ang_vel_w = self._ref_proc.compute_anchor_velocities(reference_qpos)
            anchor_lin_vel_w, anchor_ang_vel_w = self._ref_proc.apply_anchor_vel_smoothing(
                raw_anchor_lin_vel_w, raw_anchor_ang_vel_w,
            )

        # Build observation and run policy
        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=anchor_lin_vel_w,
            anchor_ang_vel_w=anchor_ang_vel_w,
            reference_window=reference_window,
        )
        obs = self._ref_proc.validate_observation(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        target_dof_pos = self._safety.clip_to_joint_limits(target_dof_pos)
        self._safety.send_positions(target_dof_pos)

        # Update state
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = reference_qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

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
            if self.remote.B.on_pressed and self._offline_playback is not None:
                logger.info("B pressed -> replaying offline motion from start")
                self._restart_offline_playback()
                return
            if self.remote.A.on_pressed:
                if self._mocap_session.state == MocapSessionState.PAUSED:
                    if self._offline_playback is not None and self._offline_playback.finished:
                        logger.info("Playback already ended; press B to replay from the start")
                    else:
                        logger.info("A pressed -> resuming playback")
                        self._resume_paused_mocap()
                else:
                    logger.info("A pressed -> pausing playback")
                    self._pause_active_mocap()
                return
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
        init_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[ROOT_DIM:FULL_QPOS_DIM] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._ref_proc.last_reference_qpos = None
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None

        # Always do a full policy reset (episode-reset semantics) to ensure
        # the TemporalCNN history is clean and action-state causality holds.
        self._reset_policy_state()

        # Kp ramp: gradually increase PD gains to avoid torque spike.
        # Unlike the old position ramp, this does NOT break action-state causality.
        self._safety.start_kp_ramp()

        self._mocap_reentry_armed = prev_mode == RobotMode.MOCAP

        self.mode = RobotMode.STANDING
        self._deactivate_dexterous_hand()
        logger.info("Mode -> STANDING (RL policy maintaining balance at default pose)")

    # ------------------------------------------------------------------
    # STANDING -> MOCAP
    # ------------------------------------------------------------------

    def _can_switch_to_mocap(self) -> bool:
        """Verify input signal is stable and values are reasonable."""
        if not self.input_provider.is_available():
            logger.warning("Mocap check: input provider not available")
            return False

        if self._offline_reference is not None:
            frame_count = min(self._check_frames, self._offline_reference.num_frames)
            valid_count = 0
            for frame_index in range(frame_count):
                try:
                    frame = self.input_provider.get_frame_by_index(frame_index)
                except (IndexError, RuntimeError, ValueError):
                    return False
                if self._ref_proc.frame_is_valid(frame):
                    valid_count += 1
                else:
                    break
            if valid_count >= frame_count:
                return True
            logger.warning("Mocap check: only %d/%d valid offline frames", valid_count, frame_count)
            return False

        has_frame = getattr(self.input_provider, "has_frame", None)
        if callable(has_frame):
            try:
                if not bool(has_frame()):
                    logger.warning("Mocap check: realtime input has no frame available yet")
                    return False
            except Exception:
                logger.warning("Mocap check: failed to query realtime input availability")
                return False

        valid_count = 0
        for _ in range(self._check_frames + 5):
            try:
                frame = self.input_provider.get_frame()
            except (TimeoutError, RuntimeError):
                return False

            if self._ref_proc.frame_is_valid(frame):
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
        init_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[ROOT_DIM:FULL_QPOS_DIM] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_commanded_motion_qpos = init_qpos.copy()
        self._mocap_reentry_armed = False

        # Full episode reset: clean policy state, alignment, timeline.
        self._reset_policy_state()

        # Reference-side interpolation: smoothly blend reference from current
        # robot state toward incoming live mocap.  This is done AFTER the
        # episode reset so the interpolator starts with a clean slate.
        self._arm_qpos_transition(init_qpos, duration_s=self._mocap_transition_duration)
        if self._offline_playback is not None:
            self._offline_playback.replay()

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
        self._deactivate_dexterous_hand()
        self._ref_proc.last_reference_qpos = None
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

    def _reset_policy_state(self) -> None:
        """Full episode-reset: clear all policy state so the TemporalCNN sees
        a clean start identical to training episode reset."""
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._qpos_interpolator.reset()
        self._reset_mocap_reference_state()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self.policy.reset()
        self.obs_builder.reset()
        self.retargeter.reset()

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
                self._ref_cfg.realtime_buffer_warmup_steps if warmup_steps is None else warmup_steps
            )
            self._reference_manager.reset()
        self._ref_proc.reset_smoothers()
        self._last_live_packet_seq = -1

    def _build_resume_alignment_qpos(self, hold_qpos: Float64Array | None, state: object) -> Float64Array:
        qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        if hold_qpos is not None:
            qpos[0:2] = np.asarray(hold_qpos, dtype=np.float64).reshape(-1)[0:2]
        base_pos = getattr(state, "base_pos", None)
        if base_pos is not None:
            qpos[0:2] = np.asarray(base_pos, dtype=np.float64).reshape(-1)[0:2]
        qpos[3:7] = np.asarray(getattr(state, "quat"), dtype=np.float64)
        qpos[ROOT_DIM:FULL_QPOS_DIM] = np.asarray(getattr(state, "qpos"), dtype=np.float64)
        return qpos

    def _restart_offline_playback(self) -> None:
        if self._offline_playback is None:
            raise RuntimeError("Offline playback replay is only available for indexed BVH input")

        state = self.robot.get_state()
        restart_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        restart_qpos[3:7] = state.quat.astype(np.float64)
        restart_qpos[ROOT_DIM:FULL_QPOS_DIM] = state.qpos.astype(np.float64)

        self._last_retarget_qpos = restart_qpos.copy()
        self._last_commanded_motion_qpos = restart_qpos.copy()
        self._offline_playback.replay()
        self._reset_policy_state()
        self._arm_qpos_transition(restart_qpos, duration_s=self._mocap_transition_duration)
        logger.info("Offline playback restarted from frame 0")

    def _hold_completed_offline_playback(self, hold_qpos: Float64Array) -> None:
        if self._offline_playback is None or self._mocap_session.state == MocapSessionState.PAUSED:
            return
        self._offline_playback.finish()
        self._mocap_session.pause(hold_qpos)
        logger.info("Offline playback reached the end; press B to replay")

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
        self._ref_proc.last_reference_qpos = hold_qpos.copy()
        self._last_commanded_motion_qpos = hold_qpos.copy()

        # Reset policy state (clears last_action, history, smoothers, etc.)
        # Note: _reset_policy_state resets _mocap_session to ACTIVE, so we
        # must call pause() *after* it to set the correct PAUSED state.
        self._reset_policy_state()
        self._mocap_session.pause(hold_qpos)
        if self._offline_playback is not None:
            self._offline_playback.pause()
        logger.info("Mocap session -> PAUSED (episode-reset)")

    def _resume_paused_mocap(self) -> None:
        if self._offline_playback is not None and self._offline_playback.finished:
            logger.info("Offline playback already ended; press B to replay from the start")
            return

        hold_qpos = self._mocap_session.hold_qpos
        if hold_qpos is None:
            raise RuntimeError("Cannot resume mocap without a paused hold qpos")
        state = self.robot.get_state()
        resume_qpos = self._build_resume_alignment_qpos(hold_qpos, state)

        self._last_commanded_motion_qpos = resume_qpos.copy()

        # Full policy reset -- clean history, zero last_action, smoothers,
        # timeline, alignment.  Also resets _mocap_session to ACTIVE.
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()

        # Override warmup steps for the resume-specific buffer warmup.
        if self._reference_manager is not None:
            self._reference_manager.set_warmup_steps(self._ref_cfg.pause_resume_warmup_steps)
            self._reference_manager.reset()

        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        if self._offline_playback is not None:
            self._last_retarget_qpos = resume_qpos.copy()
            self._arm_qpos_transition(resume_qpos, duration_s=self._pause_resume_transition_duration)
            self._offline_playback.resume()

        logger.info("Mocap session -> ACTIVE (episode-reset + reference realignment)")

    def _resolve_mocap_hold_qpos(self) -> Float64Array:
        if self._last_commanded_motion_qpos is not None:
            return self._last_commanded_motion_qpos.copy()
        if self._last_retarget_qpos is not None:
            return np.asarray(self._last_retarget_qpos, dtype=np.float64).copy()
        state = self.robot.get_state()
        hold_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        hold_qpos[3:7] = np.asarray(state.quat, dtype=np.float64)
        hold_qpos[ROOT_DIM:FULL_QPOS_DIM] = np.asarray(state.qpos, dtype=np.float64)
        return hold_qpos

    def _paused_mocap_step(self) -> None:
        hold_qpos = self._mocap_session.hold_qpos
        if hold_qpos is None:
            raise RuntimeError("Paused mocap session is missing a hold_qpos")
        self._run_static_mocap_step(hold_qpos)

    def _run_static_mocap_step(self, hold_qpos: Float64Array) -> None:
        robot_state = self.robot.get_state()
        qpos = np.asarray(hold_qpos, dtype=np.float64).copy()
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        reference_window = None
        if obs_builder_requires_reference_window(self.obs_builder):
            reference_window = build_static_reference_window(qpos, self._reference_window_builder, self.policy_hz)

        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self._ref_proc.validate_observation(obs)

        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)
        target_dof_pos = self._safety.clip_to_joint_limits(target_dof_pos)

        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()

    def _tick_dexterous_hand(self) -> None:
        active = self.mode == RobotMode.MOCAP and self._mocap_session.state == MocapSessionState.ACTIVE
        try:
            self._hand_runtime.tick(active=active)
        except Exception:
            logger.exception("Dexterous hand runtime failed; body control continues")

    def _deactivate_dexterous_hand(self) -> None:
        try:
            self._hand_runtime.tick(active=False)
        except Exception:
            logger.exception("Failed to deactivate dexterous hand runtime")

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
            self._video_runtime.stop()
        except Exception:
            pass
        try:
            self._hand_runtime.close()
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
