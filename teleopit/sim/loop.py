from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol, cast, final

import numpy as np
from numpy.typing import NDArray

from teleopit.constants import FULL_QPOS_DIM, ROOT_DIM
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.runtime.reference_config import parse_reference_config
from teleopit.inputs.realtime_packet import RealtimeInputPacket
from teleopit.interfaces import Controller, InputProvider, MessageBus, ObservationBuilder, Recorder, Retargeter, Robot, RobotState
from teleopit.sim.reference_timeline import (
    ReferenceWindow,
    ReferenceWindowBuilder,
)
from teleopit.sim.realtime_utils import RealtimeReferenceDiagnostics
from teleopit.sim.runtime_components import MotionPreparation, PolicyStepRunner, RunRecorder, RuntimePublisher, ViewerManager
from teleopit.sim.viewer_subprocess import mocap_viewer_proc, start_camera_viewer, start_robot_viewer
from teleopit.runtime.mocap_session import MocapSessionManager
from teleopit.runtime.offline_playback import OfflinePlaybackController

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


class _SupportsGet(Protocol):
    def get(self, key: str) -> object | None: ...


class SimulationMode(Enum):
    IDLE = "idle"
    STANDING = "standing"
    MOCAP = "mocap"


@final
class SimulationLoop:
    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        obs_builder: ObservationBuilder,
        bus: MessageBus,
        cfg: object,
        viewers: set[str] | None = None,
        video_runtime: object | None = None,
    ) -> None:
        self.robot: Robot = robot
        self.controller: Controller = controller
        self.obs_builder: ObservationBuilder = obs_builder
        self.bus: MessageBus = bus
        self.cfg: object = cfg
        self._video_runtime = video_runtime

        self.policy_hz: float = self._to_float(self._get_cfg("policy_hz", "sim.policy_hz", "control.policy_hz", "policy_frequency"))
        self.pd_hz: float = self._to_float(self._get_cfg("pd_hz", "sim.pd_hz", "control.pd_hz", "pd_frequency"))
        if self.policy_hz <= 0.0 or self.pd_hz <= 0.0:
            raise ValueError("policy_hz and pd_hz must be positive")
        ratio = self.pd_hz / self.policy_hz
        if ratio < 1.0:
            raise ValueError("pd_hz must be >= policy_hz")

        self.decimation: int = int(round(ratio))
        if abs(ratio - self.decimation) > 1e-6:
            raise ValueError(f"pd_hz/policy_hz must be an integer ratio, got {ratio}")

        self._num_actions: int = int(getattr(self.robot, "num_actions"))
        self._kps: Float32Array = np.asarray(getattr(self.robot, "kps"), dtype=np.float32)
        self._kds: Float32Array = np.asarray(getattr(self.robot, "kds"), dtype=np.float32)
        self._torque_limits: Float32Array = np.asarray(getattr(self.robot, "torque_limits"), dtype=np.float32)
        self._default_dof_pos: Float32Array = np.asarray(getattr(self.robot, "default_dof_pos"), dtype=np.float32)

        self._last_action: Float32Array = np.zeros((self._num_actions,), dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None
        self._realtime: bool = bool(self._try_get_cfg("realtime") or False)
        raw_debug_trace_path = self._try_get_cfg("debug_trace_path")
        self._debug_trace_path: str | None = None
        if raw_debug_trace_path not in (None, "", "null"):
            self._debug_trace_path = str(raw_debug_trace_path)

        # Motion command transition smoothing
        transition_dur = float(self._try_get_cfg("transition_duration") or 0.0)
        self._mocap_transition_duration = transition_dur
        self._pause_resume_transition_duration = float(
            self._try_get_cfg("pause_resume_transition_duration") or transition_dur
        )
        self._qpos_interpolator = QposInterpolator(transition_dur, self.policy_hz)

        self._init_reference_config()
        self._init_components(viewers)

    def _init_reference_config(self) -> None:
        """Parse reference-window / realtime-buffer configuration from self.cfg."""
        # SIM-only config keys (not in shared ReferenceConfig)
        self._playback_pause_on_end = bool(self._try_get_cfg("playback.pause_on_end", False))
        self._playback_keyboard_enabled = bool(self._try_get_cfg("playback.keyboard.enabled", False))
        self._realtime_keyboard_enabled = bool(self._try_get_cfg("keyboard.enabled", False))

        # Shared reference config (parsed once, used by both sim and sim2real)
        self._ref_cfg = parse_reference_config(self.cfg)
        rc = self._ref_cfg

        raw_reference_steps = self._try_get_cfg("reference_steps")
        self._reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / self.policy_hz,
            reference_steps=[0] if raw_reference_steps is None else cast(object, raw_reference_steps),
        )
        if not rc.retarget_buffer_enabled and self._reference_window_builder.requires_timeline:
            raise ValueError(
                "Non-zero reference_steps require retarget_buffer_enabled=true so realtime buffering "
                "can sample future/history horizons."
            )

    def _init_components(self, viewers: set[str] | None) -> None:
        """Build PolicyStepRunner, publisher, recorder helper, and viewer manager."""
        self._viewers: set[str] = set(viewers or set())
        self._step_runner = PolicyStepRunner(
            robot=self.robot,
            controller=cast(object, self.controller),
            obs_builder=self.obs_builder,
            policy_hz=self.policy_hz,
            decimation=self.decimation,
            num_actions=self._num_actions,
            kps=self._kps,
            kds=self._kds,
            torque_limits=self._torque_limits,
            default_dof_pos=self._default_dof_pos,
            qpos_interpolator=self._qpos_interpolator,
            reference_velocity_smoothing_alpha=self._ref_cfg.reference_velocity_smoothing_alpha,
            reference_anchor_velocity_smoothing_alpha=self._ref_cfg.reference_anchor_velocity_smoothing_alpha,
            reference_qpos_smoothing_alpha=self._ref_cfg.reference_qpos_smoothing_alpha,
        )
        self._publisher = RuntimePublisher(self.bus)
        self._recorder_helper = RunRecorder()
        self._viewer_manager = ViewerManager(
            robot=self.robot,
            viewers=self._viewers,
            start_robot_viewer=start_robot_viewer,
            start_camera_viewer=start_camera_viewer,
            mocap_viewer_proc=mocap_viewer_proc,
        )

    def run(
        self,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None = None,
    ) -> dict[str, float | int]:
        from teleopit.sim.session import SimLoopSession

        session = SimLoopSession(self, input_provider, retargeter, num_steps, recorder)
        return session.run()

    def run_headless(
        self,
        input_provider: InputProvider,
        retargeter: Retargeter,
        num_steps: int,
        recorder: Recorder | None = None,
    ) -> dict[str, float | int]:
        return self.run(input_provider=input_provider, retargeter=retargeter, num_steps=num_steps, recorder=recorder)

    def _compute_target_dof_pos(self, action: Float32Array) -> Float32Array:
        return self._step_runner.compute_target_dof_pos(action)

    def _build_standing_qpos(self, state: RobotState) -> Float64Array:
        standing_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        if state.base_pos is not None:
            standing_qpos[0:3] = np.asarray(state.base_pos, dtype=np.float64)[:3]
        standing_qpos[3:7] = np.asarray(state.quat, dtype=np.float64)[:4]
        standing_qpos[7:7 + self._num_actions] = self._default_dof_pos.astype(np.float64)[: self._num_actions]
        return standing_qpos

    @staticmethod
    def _drain_realtime_control_events(input_provider: InputProvider) -> tuple[object, ...]:
        pop_control_events = getattr(input_provider, "pop_control_events", None)
        if not callable(pop_control_events):
            return ()
        return tuple(pop_control_events())

    @staticmethod
    def _realtime_input_has_frame(input_provider: InputProvider) -> bool:
        has_frame = getattr(input_provider, "has_frame", None)
        if callable(has_frame):
            try:
                return bool(has_frame())
            except Exception:
                return False
        return True

    def _fetch_realtime_input_packet(
        self,
        input_provider: InputProvider,
        last_live_packet_seq: int,
    ) -> RealtimeInputPacket[dict]:
        get_realtime_input_packet = getattr(input_provider, "get_realtime_input_packet", None)
        if callable(get_realtime_input_packet):
            return cast(RealtimeInputPacket[dict], get_realtime_input_packet())

        get_packet = getattr(input_provider, "get_frame_packet", None)
        if callable(get_packet):
            frame, frame_timestamp, frame_seq = cast(tuple[dict, float, int], get_packet())
            return RealtimeInputPacket(
                frame=frame,
                timestamp_s=float(frame_timestamp),
                seq=int(frame_seq),
                control_events=(),
            )

        raise TypeError("Realtime interpolated input must provide get_frame_packet()")

    def _resolve_hold_qpos(
        self,
        last_commanded_motion_qpos: Float64Array | None,
        last_retarget_qpos: Float64Array | None,
        latest_live_retargeted: Float64Array | None,
        state: RobotState,
    ) -> Float64Array:
        if last_commanded_motion_qpos is not None:
            return last_commanded_motion_qpos.copy()
        if last_retarget_qpos is not None:
            return last_retarget_qpos.copy()
        if latest_live_retargeted is not None:
            return latest_live_retargeted.copy()
        hold_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        if state.base_pos is not None:
            hold_qpos[0:3] = np.asarray(state.base_pos, dtype=np.float64)[:3]
        hold_qpos[3:7] = np.asarray(state.quat, dtype=np.float64)[:4]
        hold_qpos[7:7 + self._num_actions] = np.asarray(state.qpos, dtype=np.float64)[: self._num_actions]
        return hold_qpos

    def _build_robot_state_qpos(self, state: RobotState) -> Float64Array:
        qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        if state.base_pos is not None:
            qpos[0:3] = np.asarray(state.base_pos, dtype=np.float64)[:3]
        qpos[3:7] = np.asarray(state.quat, dtype=np.float64)[:4]
        qpos[ROOT_DIM:ROOT_DIM + self._num_actions] = np.asarray(
            state.qpos, dtype=np.float64,
        )[: self._num_actions]
        return qpos

    def _build_resume_alignment_qpos(
        self,
        hold_qpos: Float64Array | None,
        state: RobotState,
    ) -> Float64Array:
        qpos = self._build_robot_state_qpos(state)
        if state.base_pos is None and hold_qpos is not None:
            qpos[0:2] = np.asarray(hold_qpos, dtype=np.float64).reshape(-1)[0:2]
        return qpos

    def _restart_offline_playback(
        self,
        *,
        offline_playback: OfflinePlaybackController,
        mocap_session: MocapSessionManager,
        retargeter: Retargeter,
    ) -> None:
        offline_playback.replay()
        mocap_session.reset()
        self._step_runner.reset()
        self._last_action = np.zeros((self._num_actions,), dtype=np.float32)
        self.controller.reset()
        self.obs_builder.reset()
        retargeter.reset()
        self.robot.reset()

    def _pause_offline_playback(
        self,
        *,
        offline_playback: OfflinePlaybackController,
        mocap_session: MocapSessionManager,
        hold_qpos: Float64Array,
        retargeter: Retargeter,
    ) -> None:
        offline_playback.pause()
        self._step_runner.reset()
        self._last_action = np.zeros((self._num_actions,), dtype=np.float32)
        self.controller.reset()
        self.obs_builder.reset()
        retargeter.reset()
        mocap_session.pause(hold_qpos)

    def _resume_offline_playback(
        self,
        *,
        offline_playback: OfflinePlaybackController,
        mocap_session: MocapSessionManager,
        retargeter: Retargeter,
        state: RobotState,
    ) -> None:
        if mocap_session.hold_qpos is None:
            raise RuntimeError("Cannot resume offline playback without a paused hold qpos")
        resume_qpos = self._build_robot_state_qpos(state)
        offline_playback.resume()
        mocap_session.reset()
        self._step_runner.reset()
        self._last_action = np.zeros((self._num_actions,), dtype=np.float32)
        self.controller.reset()
        self.obs_builder.reset()
        retargeter.reset()
        self._step_runner.last_retarget_qpos = resume_qpos.copy()
        self._step_runner.arm_motion_transition(
            resume_qpos,
            duration_s=self._pause_resume_transition_duration,
        )

    def _build_observation(
        self,
        state: object,
        motion_prep: object,
        last_action: Float32Array,
        reference_window: ReferenceWindow | None = None,
    ) -> Float32Array:
        return self._step_runner.build_observation(
            state,
            motion_prep,
            last_action,
            reference_window=reference_window,
        )

    def _publish(self, mimic_obs: Float32Array, action: Float32Array, robot_state: object) -> None:
        self._publisher.publish(mimic_obs, action, robot_state)

    def _record(
        self,
        recorder: Recorder | None,
        state: object,
        mimic_obs: Float32Array,
        action: Float32Array,
        target_dof_pos: Float32Array,
        torque: Float32Array,
    ) -> None:
        self._recorder_helper.record(recorder, state, mimic_obs, action, target_dof_pos, torque)

    def _retarget_to_qpos(self, retargeted: object) -> Float64Array:
        return self._step_runner._retarget_to_qpos(retargeted)

    def _get_cfg(self, *keys: str) -> object:
        for key in keys:
            value = self._try_get_cfg(key)
            if value is not None:
                return value
        raise KeyError(f"Missing required config value. Tried keys: {keys}")

    def _try_get_cfg(self, key: str, default: object | None = None) -> object | None:
        if "." in key:
            cur: object | None = self.cfg
            for part in key.split("."):
                cur = self._get_single(cur, part)
                if cur is None:
                    return default
            return default if cur is None else cur
        value = self._get_single(self.cfg, key)
        return default if value is None else value

    @staticmethod
    def _get_single(obj: object | None, key: str) -> object | None:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return cast(dict[str, object], obj).get(key)
        if hasattr(obj, "get"):
            try:
                value = cast(_SupportsGet, cast(object, obj)).get(key)
                if value is not None:
                    return value
            except Exception:
                pass
        return getattr(obj, key, None)

    @staticmethod
    def _to_float(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric config value, got {value}")
        return float(value)

    def _validate_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        return self._step_runner.validate_observation_for_policy(obs)

    def _get_root_height(self, state: RobotState) -> float:
        robot_data = getattr(self.robot, "data", None)
        if robot_data is not None:
            qpos = np.asarray(robot_data.qpos, dtype=np.float64)
            if qpos.shape[0] >= 3:
                return float(qpos[2])
        qpos_state = np.asarray(state.qpos, dtype=np.float64)
        if qpos_state.shape[0] >= 3:
            return float(qpos_state[2])
        raise ValueError("Unable to infer root height from robot state")

    def _write_debug_trace(
        self,
        debug_writer: object,
        steps_done: int,
        policy_time: float,
        frame_f: float,
        policy_obs: Float32Array,
        action: Float32Array,
        target_dof_pos: Float32Array,
        torque: Float32Array,
        preparation: MotionPreparation,
        final_state: RobotState,
        reference_window: ReferenceWindow | None,
        reference_timeline: object | None,
        realtime_reference_diag: object | None,
    ) -> None:
        controller_debug_inputs: dict[str, object] = {}
        get_debug_inputs = getattr(self.controller, "get_debug_inputs", None)
        if callable(get_debug_inputs):
            controller_debug_inputs = cast(dict[str, object], get_debug_inputs())
        add_step = getattr(debug_writer, "add_step")
        add_step(
            step=np.int64(steps_done),
            policy_time=np.float64(policy_time),
            frame_f=np.float64(frame_f),
            obs=np.asarray(policy_obs, dtype=np.float32),
            obs_history=controller_debug_inputs.get("obs_history"),
            action=np.asarray(action, dtype=np.float32),
            target_dof_pos=np.asarray(target_dof_pos, dtype=np.float32),
            motion_qpos=np.asarray(preparation.qpos[: 7 + self._num_actions], dtype=np.float32),
            motion_joint_vel=np.asarray(preparation.raw_motion_joint_vel, dtype=np.float32),
            smoothed_motion_joint_vel=np.asarray(preparation.motion_joint_vel, dtype=np.float32),
            motion_anchor_lin_vel_w=preparation.raw_motion_anchor_lin_vel_w,
            motion_anchor_ang_vel_w=preparation.raw_motion_anchor_ang_vel_w,
            smoothed_motion_anchor_lin_vel_w=preparation.motion_anchor_lin_vel_w,
            smoothed_motion_anchor_ang_vel_w=preparation.motion_anchor_ang_vel_w,
            robot_qpos=np.asarray(final_state.qpos, dtype=np.float32),
            robot_qvel=np.asarray(final_state.qvel, dtype=np.float32),
            robot_quat=np.asarray(final_state.quat, dtype=np.float32),
            robot_base_pos=(None if final_state.base_pos is None else np.asarray(final_state.base_pos, dtype=np.float32)),
            torque=np.asarray(torque, dtype=np.float32),
            reference_base_time_s=(None if reference_window is None else np.asarray(reference_window.base_time_s, dtype=np.float64)),
            reference_steps=(None if reference_window is None else np.asarray(reference_window.reference_steps, dtype=np.int64)),
            reference_sample_modes=(None if reference_window is None else np.asarray(reference_window.modes(), dtype=np.str_)),
            reference_sample_alphas=(None if reference_window is None else np.asarray(reference_window.alphas(), dtype=np.float32)),
            reference_sample_used_fallback=(None if reference_window is None else np.asarray(reference_window.fallback_mask(), dtype=np.bool_)),
            reference_sample_timestamps=(None if reference_window is None else np.asarray(reference_window.timestamps(), dtype=np.float64)),
            reference_buffer_len=(None if reference_timeline is None else np.asarray(len(reference_timeline), dtype=np.int64)),  # type: ignore[arg-type]
            reference_future_horizon_steps=(None if realtime_reference_diag is None else np.asarray(getattr(realtime_reference_diag, "future_horizon_steps"), dtype=np.int64)),
            reference_real_frame_count=(None if realtime_reference_diag is None else np.asarray(getattr(realtime_reference_diag, "real_frame_count"), dtype=np.int64)),
            reference_warmup_done=(None if realtime_reference_diag is None else np.asarray(getattr(realtime_reference_diag, "warmup_done"), dtype=np.bool_)),
            reference_used_repeat_padding=(None if realtime_reference_diag is None else np.asarray(getattr(realtime_reference_diag, "used_repeat_padding"), dtype=np.bool_)),
            reference_padding_active=(None if realtime_reference_diag is None else np.asarray(getattr(realtime_reference_diag, "padding_active"), dtype=np.bool_)),
        )

    def _log_reference_window(self, reference_window: ReferenceWindow, buffer_len: int) -> None:
        import logging

        logging.getLogger(__name__).warning(
            "Reference timeline fallback | buffer_len=%d | base_time=%.6f | steps=%s | modes=%s",
            buffer_len,
            reference_window.base_time_s,
            list(reference_window.reference_steps),
            list(reference_window.modes()),
        )

    def _log_repeat_padding(
        self,
        reference_window: ReferenceWindow,
        diagnostics: RealtimeReferenceDiagnostics,
        buffer_len: int,
    ) -> None:
        import logging

        logging.getLogger(__name__).warning(
            "Reference timeline repeat padding | buffer_len=%d | future_horizon_steps=%d | steps=%s",
            buffer_len,
            diagnostics.future_horizon_steps,
            list(reference_window.reference_steps),
        )
