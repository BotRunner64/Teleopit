"""Per-run session state for SimulationLoop.

Encapsulates the ~30 local variables and 5 closures that previously lived
inside ``SimulationLoop.run()``, converting them to instance attributes
and methods.  ``SimulationLoop.run()`` delegates here:

    session = SimLoopSession(self, ...)
    return session.run()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from teleopit.debug.rollout_trace import RolloutTraceWriter
from teleopit.interfaces import InputProvider, Recorder, Retargeter, RobotState
from teleopit.sim.reference_motion import (
    OfflineReferenceMotion,
    interpolate_human_frames,
    interpolate_retarget_qpos,
)
from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindow
from teleopit.sim.reference_utils import (
    build_offline_reference_window,
    build_static_reference_window,
    obs_builder_requires_reference_window,
)
from teleopit.sim.realtime_utils import RealtimeReferenceDiagnostics, RealtimeReferenceManager
from teleopit.sim.runtime_components import MotionPreparation
from teleopit.runtime.arm_mocap import compose_arm_reference_window
from teleopit.runtime.mocap_session import MocapSessionManager, MocapSessionState
from teleopit.runtime.offline_playback import OfflinePlaybackController
from teleopit.runtime.terminal_keyboard import TerminalKeyboardReader
from teleopit.inputs.realtime_packet import ControlEventType

if TYPE_CHECKING:
    from teleopit.sim.loop import SimulationLoop, SimulationMode

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
_logger = logging.getLogger(__name__)


class SimLoopSession:
    """Encapsulates per-run state for a single simulation session.

    Created and run by :meth:`SimulationLoop.run`.  All state that was
    formerly 30+ local variables and 5 nested closures now lives here.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        loop: SimulationLoop,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None,
    ) -> None:
        self._loop = loop
        self._input_provider = input_provider
        self._retargeter = retargeter
        self._recorder = recorder

        # Convenience aliases for heavily-used loop attributes
        self._step_runner = loop._step_runner
        self._viewer_manager = loop._viewer_manager

        # Per-run scalars
        self.steps_done: int = 0
        policy_dt = 1.0 / loop.policy_hz
        self.policy_dt: float = policy_dt
        has_viewers = self._viewer_manager.has_viewers()
        self.has_viewers: bool = has_viewers
        self.needs_pacing: bool = has_viewers or loop._realtime
        self.wall_start: float = time.monotonic() if self.needs_pacing else 0.0
        self.max_steps: int = num_steps if num_steps > 0 else 2**63

        # Input detection
        input_fps: float = float(getattr(input_provider, "fps", loop.policy_hz))
        self.offline_reference: OfflineReferenceMotion | None = None
        self.offline_playback: OfflinePlaybackController | None = None
        if hasattr(input_provider, "__len__") and hasattr(input_provider, "get_frame_by_index"):
            self.offline_reference = OfflineReferenceMotion(input_provider, retargeter)
            input_fps = self.offline_reference.fps
            self.offline_playback = OfflinePlaybackController(
                duration_s=self.offline_reference.duration_s,
                step_dt_s=policy_dt,
                pause_on_end=loop._playback_pause_on_end,
            )
        self.input_fps: float = input_fps

        has_realtime_packet_input = self.offline_reference is None and (
            callable(getattr(input_provider, "get_realtime_input_packet", None))
            or callable(getattr(input_provider, "get_frame_packet", None))
        )
        has_realtime_keyboard_input = (
            self.offline_reference is None
            and loop._realtime_keyboard_enabled
            and (
                callable(getattr(input_provider, "has_frame", None))
                or callable(getattr(input_provider, "pop_control_events", None))
            )
        )
        self.realtime_packet_input: bool = has_realtime_packet_input
        self.realtime_interpolated_input: bool = has_realtime_packet_input or has_realtime_keyboard_input

        if loop._playback_keyboard_enabled and self.offline_reference is None and loop._try_get_cfg("playback.keyboard.enabled") is True:
            raise ValueError("playback.keyboard.enabled requires an offline BVH input provider.")

        ref_cfg = loop._ref_cfg
        self.realtime_input_delay_s: float = (
            1.0 / input_fps
            if self.realtime_packet_input and ref_cfg.reference_delay_s is None
            else float(ref_cfg.reference_delay_s or 0.0)
        )

        if (
            loop._reference_window_builder.requires_timeline
            and not self.realtime_packet_input
            and self.offline_reference is None
        ):
            raise ValueError(
                "Non-zero reference_steps require either a realtime input provider exposing "
                "get_frame_packet() or an offline input provider with indexed frame access. "
                "Current-only input paths cannot provide future/history windows."
            )

        # Reference timeline + manager
        self.reference_timeline: ReferenceTimeline | None = None
        if self.realtime_packet_input and ref_cfg.retarget_buffer_enabled:
            loop._reference_window_builder.validate_runtime_support(
                delay_s=self.realtime_input_delay_s,
                window_s=ref_cfg.retarget_buffer_window_s,
                config_label="SimulationLoop reference timeline",
            )
            self.reference_timeline = ReferenceTimeline(window_s=ref_cfg.retarget_buffer_window_s)

        self.realtime_reference_manager: RealtimeReferenceManager | None = None
        if self.reference_timeline is not None:
            self.realtime_reference_manager = RealtimeReferenceManager(
                reference_window_builder=loop._reference_window_builder,
                warmup_steps=ref_cfg.realtime_buffer_warmup_steps,
            )

        # Realtime live-frame tracking
        self.last_live_packet_seq: int = -1
        self.previous_live_human_frame: dict | None = None
        self.previous_live_retargeted: Float64Array | None = None
        self.previous_live_timestamp: float | None = None
        self.latest_live_human_frame: dict | None = None
        self.latest_live_retargeted: Float64Array | None = None
        self.latest_live_timestamp: float | None = None

        # Mocap session + motion state
        self.mocap_session: MocapSessionManager = MocapSessionManager()
        self.last_commanded_motion_qpos: Float64Array | None = None

        # Frame cache (updated each iteration)
        self.last_bvh_idx: int = -1
        self.cached_human_frame: dict | None = None
        self.cached_retargeted: object = None

        # Keyboard / mode control
        self.playback_stop_requested: bool = False
        keyboard_reader: TerminalKeyboardReader | None = None
        if loop._playback_keyboard_enabled and self.offline_reference is not None:
            keyboard_reader = TerminalKeyboardReader()
        elif loop._realtime_keyboard_enabled and self.realtime_interpolated_input:
            keyboard_reader = TerminalKeyboardReader()
        if keyboard_reader is not None and not keyboard_reader.active:
            keyboard_reader.close()
            keyboard_reader = None
        self.keyboard_reader: TerminalKeyboardReader | None = keyboard_reader

        self.realtime_keyboard_mode_enabled: bool = bool(
            self.realtime_interpolated_input
            and loop._realtime_keyboard_enabled
            and self.keyboard_reader is not None
            and self.keyboard_reader.active
        )

        from teleopit.sim.loop import SimulationMode
        self.simulation_mode: SimulationMode = (
            SimulationMode.STANDING if self.realtime_keyboard_mode_enabled else SimulationMode.MOCAP
        )
        if self.simulation_mode == SimulationMode.STANDING:
            loop._set_standing_reference(loop.robot.get_state())

        # Debug writer
        self.debug_writer: RolloutTraceWriter | None = None
        if loop._debug_trace_path is not None:
            self.debug_writer = RolloutTraceWriter(
                Path(loop._debug_trace_path),
                metadata={
                    "source": "sim2sim",
                    "policy_hz": loop.policy_hz,
                    "pd_hz": loop.pd_hz,
                    "input_fps": self.input_fps,
                    "reference_steps": list(loop._reference_window_builder.reference_steps),
                },
            )

    # ------------------------------------------------------------------
    # State-management methods (formerly closures with nonlocal)
    # ------------------------------------------------------------------

    def reset_runtime_tracking(self) -> None:
        ref_cfg = self._loop._ref_cfg
        if self.reference_timeline is not None:
            self.reference_timeline.clear()
        if self.realtime_reference_manager is not None:
            self.realtime_reference_manager.set_warmup_steps(ref_cfg.realtime_buffer_warmup_steps)
            self.realtime_reference_manager.reset()
        self.previous_live_human_frame = None
        self.previous_live_retargeted = None
        self.previous_live_timestamp = None
        self.latest_live_human_frame = None
        self.latest_live_retargeted = None
        self.latest_live_timestamp = None
        self.last_live_packet_seq = -1
        self.cached_human_frame = None
        self.cached_retargeted = None

    def reset_policy_reference_state(self, *, reset_mocap_session: bool = True) -> None:
        self._step_runner.reset()
        self._loop.controller.reset()
        self._loop.obs_builder.reset()
        if reset_mocap_session:
            self.mocap_session.reset()
        self.last_commanded_motion_qpos = None
        self.reset_runtime_tracking()

    def enter_standing_mode(self) -> None:
        from teleopit.sim.loop import SimulationMode
        self.reset_policy_reference_state()
        self._loop._set_standing_reference(self._loop.robot.get_state())
        self.simulation_mode = SimulationMode.STANDING

    def enter_mocap_mode(self) -> None:
        from teleopit.sim.loop import SimulationMode
        loop = self._loop
        if not loop._realtime_input_has_frame(self._input_provider):
            _logger.warning("Cannot switch to MOCAP yet: realtime input has no frame available")
            return
        state = loop.robot.get_state()
        start_qpos = loop._resolve_hold_qpos(None, None, None, state)
        self.reset_policy_reference_state()
        self._step_runner.last_retarget_qpos = start_qpos.copy()
        self.last_commanded_motion_qpos = start_qpos.copy()
        self.simulation_mode = SimulationMode.MOCAP

    def toggle_arms_mode(self) -> None:
        from teleopit.sim.loop import SimulationMode
        if not self.realtime_interpolated_input or self.simulation_mode not in (SimulationMode.MOCAP, SimulationMode.ARMS):
            return
        if self.mocap_session.state == MocapSessionState.PAUSED:
            _logger.info("Ignoring arm-only mode toggle while mocap session is paused")
            return
        loop = self._loop
        state = loop.robot.get_state()
        resume_qpos = loop._build_resume_alignment_qpos(self.last_commanded_motion_qpos, state)
        if self.simulation_mode == SimulationMode.MOCAP:
            loop._set_standing_reference(state)
            self.simulation_mode = SimulationMode.ARMS
        else:
            self.simulation_mode = SimulationMode.MOCAP
        self._step_runner.reset()
        loop.controller.reset()
        loop.obs_builder.reset()
        self.mocap_session.reset()
        self.last_commanded_motion_qpos = None
        self._step_runner.reset_reference_alignment(resume_qpos)
        self.last_commanded_motion_qpos = resume_qpos.copy()
        _logger.info("Simulation mode -> %s", self.simulation_mode.value.upper())

    def toggle_realtime_mocap_pause(self) -> None:
        loop = self._loop
        if self.mocap_session.state == MocapSessionState.PAUSED:
            hold_qpos = self.mocap_session.hold_qpos
            if hold_qpos is None:
                raise RuntimeError("Cannot resume mocap without a paused hold qpos")
            resume_qpos = loop._build_resume_alignment_qpos(hold_qpos, loop.robot.get_state())
            self.reset_policy_reference_state()
            self._step_runner.reset_reference_alignment(resume_qpos)
            self.last_commanded_motion_qpos = resume_qpos.copy()
            return
        hold_qpos = loop._resolve_hold_qpos(
            self.last_commanded_motion_qpos,
            self._step_runner.last_retarget_qpos,
            self.latest_live_retargeted,
            loop.robot.get_state(),
        )
        self.reset_policy_reference_state()
        self.mocap_session.pause(hold_qpos)
        self.last_commanded_motion_qpos = hold_qpos.copy()

    # ------------------------------------------------------------------
    # Keyboard handling
    # ------------------------------------------------------------------

    def _handle_realtime_keyboard(self) -> bool:
        """Handle keyboard events in realtime keyboard mode. Returns True to break the main loop."""
        from teleopit.sim.loop import SimulationMode
        assert self.keyboard_reader is not None
        for key_event in self.keyboard_reader.poll():
            key = key_event.key.lower()
            if key == "q":
                self.playback_stop_requested = True
                return True
            if self.simulation_mode == SimulationMode.STANDING:
                if key == "y":
                    self.enter_mocap_mode()
                continue
            if key == "x":
                self.enter_standing_mode()
                continue
            if key == "b":
                self.toggle_arms_mode()
                continue
            if key == "a":
                self.toggle_realtime_mocap_pause()
        return False

    def _handle_offline_keyboard(self) -> bool:
        """Handle keyboard events in offline playback mode. Returns True to break the main loop."""
        assert self.keyboard_reader is not None
        assert self.offline_playback is not None
        loop = self._loop
        for key_event in self.keyboard_reader.poll():
            key = key_event.key.lower()
            if key == "q":
                self.playback_stop_requested = True
                return True
            if key == "r":
                loop._restart_offline_playback(
                    offline_playback=self.offline_playback,
                    mocap_session=self.mocap_session,
                )
                self.cached_human_frame = None
                self.cached_retargeted = None
                self.last_commanded_motion_qpos = None
                continue
            if key not in (" ", "p"):
                continue
            if self.mocap_session.state == MocapSessionState.PAUSED:
                if self.offline_playback.finished:
                    logging.getLogger(__name__).info(
                        "Offline playback already ended; press r to replay from frame 0."
                    )
                else:
                    loop._resume_offline_playback(
                        offline_playback=self.offline_playback,
                        mocap_session=self.mocap_session,
                        state=loop.robot.get_state(),
                    )
                    self.last_commanded_motion_qpos = None
            else:
                hold_qpos = loop._resolve_hold_qpos(
                    self.last_commanded_motion_qpos,
                    self._step_runner.last_retarget_qpos,
                    None,
                    loop.robot.get_state(),
                )
                loop._pause_offline_playback(
                    offline_playback=self.offline_playback,
                    mocap_session=self.mocap_session,
                    hold_qpos=hold_qpos,
                )
        return False

    # ------------------------------------------------------------------
    # Input fetching
    # ------------------------------------------------------------------

    def _fetch_standing_input(self) -> tuple[bool, ReferenceWindow | None, RealtimeReferenceDiagnostics | None]:
        """Fetch input when in STANDING mode (keyboard). Returns (new_bvh_frame, ref_window, diag)."""
        self.cached_human_frame = None
        if self._loop._standing_qpos is None:
            self.cached_retargeted = self._loop._set_standing_reference(self._loop.robot.get_state())
        else:
            self.cached_retargeted = self._loop._standing_qpos.copy()
        return False, None, None

    def _fetch_offline_reference_input(
        self, policy_time: float,
    ) -> tuple[bool, ReferenceWindow | None, RealtimeReferenceDiagnostics | None, bool]:
        """Fetch input from offline reference. Returns (new_bvh_frame, ref_window, diag, should_break)."""
        loop = self._loop
        assert self.offline_playback is not None
        assert self.offline_reference is not None

        if self.mocap_session.state == MocapSessionState.PAUSED:
            hold_qpos = self.mocap_session.hold_qpos
            if hold_qpos is None:
                raise RuntimeError("Paused offline playback is missing a hold pose")
            self.cached_retargeted = hold_qpos.copy()
            return False, None, None, False

        sampled = self.offline_reference.sample(policy_time)
        if sampled is None:
            if self.offline_playback.pause_on_end:
                self.offline_playback.finish()
                hold_qpos = loop._resolve_hold_qpos(
                    self.last_commanded_motion_qpos,
                    self._step_runner.last_retarget_qpos,
                    None,
                    loop.robot.get_state(),
                )
                self.mocap_session.pause(hold_qpos)
                self.cached_retargeted = hold_qpos.copy()
                return False, None, None, False
            else:
                return False, None, None, True  # should_break

        reference_window: ReferenceWindow | None = None
        if obs_builder_requires_reference_window(loop.obs_builder):
            reference_window = build_offline_reference_window(
                self.offline_reference, policy_time,
                loop._reference_window_builder, loop.policy_hz,
            )
        self.cached_human_frame = sampled.human_frame
        self.cached_retargeted = sampled.qpos
        self.last_bvh_idx = sampled.frame_idx0
        return True, reference_window, None, False

    def _fetch_realtime_input(self) -> tuple[bool, ReferenceWindow | None, RealtimeReferenceDiagnostics | None, bool]:
        """Fetch input from realtime provider. Returns (new_bvh_frame, ref_window, diag, should_continue)."""
        loop = self._loop
        packet = loop._fetch_realtime_input_packet(self._input_provider, self.last_live_packet_seq)
        human_frame = cast(dict, packet.frame)
        frame_timestamp = float(packet.timestamp_s)
        frame_seq = int(packet.seq)
        for control_event in packet.control_events:
            if control_event.event_type == ControlEventType.TOGGLE_ARMS:
                self.toggle_arms_mode()
                continue
            if control_event.event_type == ControlEventType.TOGGLE_PAUSE:
                self.toggle_realtime_mocap_pause()
        new_bvh_frame = frame_seq != self.last_live_packet_seq

        if self.mocap_session.state == MocapSessionState.PAUSED:
            self.cached_human_frame = human_frame
            self.cached_retargeted = self.mocap_session.hold_qpos
            if self.cached_retargeted is None:
                raise RuntimeError("Paused mocap session is missing a hold pose")
            return new_bvh_frame, None, None, False

        if new_bvh_frame:
            self.previous_live_human_frame = self.latest_live_human_frame
            self.previous_live_timestamp = self.latest_live_timestamp
            self.latest_live_human_frame = human_frame
            retargeted_qpos = self._step_runner._retarget_to_qpos(self._retargeter.retarget(human_frame))
            if self.reference_timeline is not None:
                self.reference_timeline.append(retargeted_qpos, float(frame_timestamp))
                if self.realtime_reference_manager is not None:
                    self.realtime_reference_manager.note_realtime_frame()
            else:
                self.previous_live_retargeted = self.latest_live_retargeted
                self.latest_live_retargeted = retargeted_qpos
            self.latest_live_timestamp = float(frame_timestamp)
            self.last_live_packet_seq = int(frame_seq)

        if self.latest_live_human_frame is None:
            raise RuntimeError("Realtime input did not provide an initial frame")

        target_base_time = time.monotonic() - self.realtime_input_delay_s
        if (
            self.previous_live_human_frame is not None
            and self.previous_live_timestamp is not None
            and self.latest_live_timestamp is not None
            and self.latest_live_timestamp > self.previous_live_timestamp + 1e-6
        ):
            alpha = (target_base_time - self.previous_live_timestamp) / (
                self.latest_live_timestamp - self.previous_live_timestamp
            )
            alpha = float(np.clip(alpha, 0.0, 1.0))
            self.cached_human_frame = interpolate_human_frames(
                self.previous_live_human_frame,
                self.latest_live_human_frame,
                alpha,
            )
        else:
            self.cached_human_frame = self.latest_live_human_frame

        reference_window: ReferenceWindow | None = None
        realtime_reference_diag: RealtimeReferenceDiagnostics | None = None

        if self.reference_timeline is not None:
            if self.realtime_reference_manager is None:
                raise RuntimeError("Realtime reference manager must be initialized when using reference_timeline")
            if not self.realtime_reference_manager.warmup_done:
                time.sleep(min(self.policy_dt, 1.0 / max(self.input_fps, 1.0)))
                return new_bvh_frame, None, None, True  # should_continue
            reference_window, realtime_reference_diag = self.realtime_reference_manager.sample(
                self.reference_timeline,
                target_base_time,
            )
            if loop._ref_cfg.reference_debug_log and any(reference_window.fallback_mask()):
                loop._log_reference_window(reference_window, len(self.reference_timeline))
            self.cached_retargeted = reference_window.current_sample().qpos
        else:
            if self.latest_live_retargeted is None:
                raise RuntimeError("Realtime input did not provide an initial retargeted frame")
            if (
                self.previous_live_retargeted is not None
                and self.previous_live_timestamp is not None
                and self.latest_live_timestamp is not None
                and self.latest_live_timestamp > self.previous_live_timestamp + 1e-6
            ):
                alpha = (target_base_time - self.previous_live_timestamp) / (
                    self.latest_live_timestamp - self.previous_live_timestamp
                )
                alpha = float(np.clip(alpha, 0.0, 1.0))
                self.cached_retargeted = interpolate_retarget_qpos(
                    self.previous_live_retargeted,
                    self.latest_live_retargeted,
                    alpha,
                )
            else:
                self.cached_retargeted = self.latest_live_retargeted

        from teleopit.sim.loop import SimulationMode
        if self.simulation_mode == SimulationMode.ARMS:
            self.cached_retargeted = loop._compose_arm_reference(cast(Float64Array, self.cached_retargeted))
            if reference_window is not None:
                assert loop._standing_qpos is not None
                reference_window = compose_arm_reference_window(
                    reference_window,
                    standing_qpos=loop._standing_qpos,
                    arm_joint_indices=loop._arm_joint_indices,
                    num_actions=loop._num_actions,
                )

        return new_bvh_frame, reference_window, realtime_reference_diag, False

    def _fetch_simple_bvh_input(self, frame_f: float) -> tuple[bool, bool]:
        """Fetch input from simple BVH (non-offline-reference) provider. Returns (new_bvh_frame, should_break)."""
        bvh_idx = int(frame_f)
        new_bvh_frame = bvh_idx != self.last_bvh_idx
        if new_bvh_frame:
            if not self._input_provider.is_available():
                return new_bvh_frame, True  # should_break
            self.cached_human_frame = self._input_provider.get_frame()
            self.cached_retargeted = self._retargeter.retarget(self.cached_human_frame)
            self.last_bvh_idx = bvh_idx
        return new_bvh_frame, False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, float | int]:
        """Execute the simulation loop. Returns summary dict."""
        from teleopit.sim.loop import SimulationMode

        loop = self._loop
        self._step_runner.reset()
        self._viewer_manager.ensure_mocap_viewer(cast(object, self._input_provider))
        self._viewer_manager.wait_until_ready(timeout_s=10.0)

        try:
            if loop._video_runtime is not None:
                loop._video_runtime.start()
            while self.steps_done < self.max_steps:
                if self.has_viewers and not self._viewer_manager.any_active():
                    break

                # --- Keyboard handling ---
                if self.keyboard_reader is not None:
                    if self.offline_playback is None:
                        if self.realtime_keyboard_mode_enabled:
                            if self._handle_realtime_keyboard():
                                break
                        else:
                            raise RuntimeError("Keyboard playback polling requires an offline playback controller")
                    else:
                        if self._handle_offline_keyboard():
                            break
                        if self.playback_stop_requested:
                            break

                if self.realtime_keyboard_mode_enabled and self.simulation_mode == SimulationMode.STANDING:
                    loop._drain_realtime_control_events(self._input_provider)

                # --- Compute time/frame ---
                policy_time = self.steps_done * self.policy_dt
                if self.offline_playback is not None:
                    policy_time = self.offline_playback.current_time_s
                frame_f = policy_time * self.input_fps

                # --- Fetch input ---
                reference_window: ReferenceWindow | None = None
                realtime_reference_diag: RealtimeReferenceDiagnostics | None = None
                new_bvh_frame = False

                if self.realtime_keyboard_mode_enabled and self.simulation_mode == SimulationMode.STANDING:
                    new_bvh_frame, reference_window, realtime_reference_diag = self._fetch_standing_input()
                elif self.offline_reference is not None:
                    if self.offline_playback is None:
                        raise RuntimeError("Offline playback controller must be initialized for offline references")
                    new_bvh_frame, reference_window, realtime_reference_diag, should_break = (
                        self._fetch_offline_reference_input(policy_time)
                    )
                    if should_break:
                        break
                elif self.realtime_interpolated_input:
                    new_bvh_frame, reference_window, realtime_reference_diag, should_continue = (
                        self._fetch_realtime_input()
                    )
                    if should_continue:
                        continue
                else:
                    new_bvh_frame, should_break = self._fetch_simple_bvh_input(frame_f)
                    if should_break:
                        break

                # --- Policy step ---
                state = loop.robot.get_state()
                if self.realtime_keyboard_mode_enabled and self.simulation_mode == SimulationMode.STANDING:
                    preparation = self._step_runner.prepare_static_motion_command(self.cached_retargeted)
                    if obs_builder_requires_reference_window(loop.obs_builder):
                        reference_window = build_static_reference_window(
                            self.cached_retargeted,
                            loop._reference_window_builder,
                            loop.policy_hz,
                        )
                elif self.mocap_session.state == MocapSessionState.PAUSED:
                    hold_qpos = self.mocap_session.hold_qpos
                    if hold_qpos is None:
                        raise RuntimeError("Paused mocap session is missing a hold pose")
                    preparation = self._step_runner.prepare_static_motion_command(hold_qpos)
                    if obs_builder_requires_reference_window(loop.obs_builder):
                        reference_window = build_static_reference_window(
                            hold_qpos, loop._reference_window_builder, loop.policy_hz,
                        )
                else:
                    preparation = self._step_runner.prepare_motion_command(self.cached_retargeted, state)

                obs = self._step_runner.build_observation(
                    state, preparation, self._step_runner.last_action, reference_window=reference_window,
                )
                policy_obs = self._step_runner.validate_observation_for_policy(obs)
                action: Float32Array = np.asarray(
                    loop.controller.compute_action(policy_obs), dtype=np.float32,
                ).reshape(-1)
                if action.shape[0] != loop._num_actions:
                    raise ValueError(f"Controller returned {action.shape[0]} actions, expected {loop._num_actions}")

                target_dof_pos = self._step_runner.compute_target_dof_pos(action)
                torque, final_state = self._step_runner.apply_control(target_dof_pos)
                loop._publisher.publish(preparation.mimic_obs, action, final_state)
                loop._recorder_helper.record(self._recorder, final_state, preparation.mimic_obs, action, target_dof_pos, torque)
                self._viewer_manager.write_sim2sim(loop.robot)
                self._viewer_manager.write_camera(loop.robot)
                if loop._video_runtime is not None:
                    loop._video_runtime.tick()
                self._viewer_manager.write_retarget(preparation.retarget_viewer_qpos)
                if self.cached_human_frame is not None and (
                    self.offline_reference is not None or new_bvh_frame or self.realtime_interpolated_input
                ):
                    self._viewer_manager.write_mocap(cast(object, self._input_provider), self.cached_human_frame)

                if self.debug_writer is not None:
                    loop._write_debug_trace(
                        debug_writer=self.debug_writer,
                        steps_done=self.steps_done,
                        policy_time=policy_time,
                        frame_f=frame_f,
                        policy_obs=policy_obs,
                        action=action,
                        target_dof_pos=target_dof_pos,
                        torque=torque,
                        preparation=preparation,
                        final_state=final_state,
                        reference_window=reference_window,
                        reference_timeline=self.reference_timeline,
                        realtime_reference_diag=realtime_reference_diag,
                    )

                # Real-time pacing
                if self.needs_pacing:
                    sim_time = (self.steps_done + 1) * self.policy_dt
                    wall_elapsed = time.monotonic() - self.wall_start
                    sleep_time = sim_time - wall_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self._step_runner.finish_step(action, preparation.qpos)
                self.last_commanded_motion_qpos = preparation.qpos.copy()
                if (
                    self.offline_playback is not None
                    and self.mocap_session.state != MocapSessionState.PAUSED
                ):
                    if self.offline_playback.advance():
                        if self.offline_playback.pause_on_end:
                            self.mocap_session.pause(preparation.qpos.copy())
                        else:
                            self.steps_done += 1
                            break
                self.steps_done += 1
        except KeyboardInterrupt:
            pass
        finally:
            if loop._video_runtime is not None:
                loop._video_runtime.stop()
            self._viewer_manager.shutdown()
            if self.keyboard_reader is not None:
                self.keyboard_reader.close()
            if self.debug_writer is not None:
                self.debug_writer.save()

        final_state = loop.robot.get_state()
        return {
            "steps": self.steps_done,
            "root_height": loop._get_root_height(final_state),
            "policy_hz": loop.policy_hz,
            "pd_hz": loop.pd_hz,
            "decimation": loop.decimation,
            "playback_stop": int(self.playback_stop_requested),
        }
