"""Multiprocess sim2real runtime using ZMQ and shared memory."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent
from enum import Enum
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS, ROOT_DIM
from teleopit.controllers.observation import VelCmdObservationBuilder, align_motion_qpos_yaw
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.bvh_provider import BVHInputProvider
from teleopit.inputs.human_frame_validation import validate_human_frame
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.pico_video import PicoVideoRuntime, bridge_video_source, parse_pico_video_config
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.offline_playback import OfflinePlaybackController
from teleopit.runtime.common import cfg_get, parse_viewers, require_section
from teleopit.runtime.console import (
    OPERATOR_LOGGER_NAME,
    PlainConsole,
    configure_runtime_logging,
    console_show_timing,
    console_timing_interval_s,
    sim2real_keyboard_controls,
)
from teleopit.runtime.factory import _build_policy_components, build_simulation_cfg
from teleopit.runtime.arm_mocap import (
    compose_arm_reference,
    compose_arm_reference_window,
    parse_arm_joint_indices,
)
from teleopit.runtime.mocap_session import MocapSessionManager, MocapSessionState
from teleopit.runtime.reference_config import parse_reference_config
from teleopit.runtime.terminal_keyboard import TerminalKeyboardReader
from teleopit.recording.hdf5 import (
    build_mode_observation,
    build_observation_state,
    normalize_hand_action,
    normalize_action_reference_qpos,
)
from teleopit.sim.reference_motion import OfflineReferenceMotion
from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindow, ReferenceWindowBuilder
from teleopit.sim.reference_utils import (
    build_offline_reference_window,
    build_static_reference_window,
    obs_builder_requires_reference_window,
)
from teleopit.sim.realtime_utils import RealtimeReferenceManager
from teleopit.sim.viewer_subprocess import start_robot_viewer
from teleopit.sim2real.hands.worker import build_hand_runtime
from teleopit.sim2real.hands.base import HandPoseCommand
from teleopit.sim2real.hands.linkerhand_l6 import parse_linkerhand_l6_config
from teleopit.sim2real.hands.linkerhand_o6 import parse_linkerhand_o6_config
from teleopit.sim2real.mp.ipc import (
    BODY_TOPIC,
    COMMAND_TOPIC,
    CONTROL_EVENTS_TOPIC,
    CONTROLLER_TOPIC,
    HAND_COMMAND_TOPIC,
    HAND_TOPIC,
    HEALTH_TOPIC,
    MODE_TOPIC,
    RECORD_TOPIC,
    REFERENCE_TOPIC,
    VIDEO_TOPIC,
    LatestSubscriber,
    Sim2RealIpcEndpoints,
    ZmqPublisher,
    default_endpoints,
)
from teleopit.sim2real.mp.messages import (
    BodyFramePacket,
    CommandPacket,
    ControlEventsPacket,
    HandCommandPacket,
    HealthPacket,
    ModeStatePacket,
    ReferencePacket,
    RecordStepPacket,
    SnapshotPacket,
    SharedFrameDescriptor,
)
from teleopit.sim2real.mp.shm import SharedFrameRingReader, SharedFrameRingWriter
from teleopit.sim2real.reference_processor import Sim2RealReferenceProcessor
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.safety import Sim2RealSafetyManager
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - OmegaConf is a project dependency.
    OmegaConf = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
operator_logger = logging.getLogger(OPERATOR_LOGGER_NAME)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class RobotMode(Enum):
    IDLE = "idle"
    STANDING = "standing"
    MOCAP = "mocap"
    ARMS = "arms"
    DAMPING = "damping"


class _LoopTimingReporter:
    def __init__(
        self,
        *,
        target_period_s: float,
        log_interval_s: float = 1.0,
        deadline_miss_tolerance_s: float = 0.001,
        enabled: bool = True,
    ) -> None:
        self._target_period_s = float(target_period_s)
        self._log_interval_s = float(log_interval_s)
        self._deadline_miss_tolerance_s = float(deadline_miss_tolerance_s)
        self._enabled = bool(enabled)
        self._window_start_s: float | None = None
        self._loop_ms: list[float] = []
        self._late_ms: list[float] = []
        self._work_ms: list[float] = []
        self._pico_age_ms: list[float] = []
        self._deadline_miss_count = 0
        self._work_overrun_count = 0

    def record(self, *, loop_start_s: float, work_elapsed_s: float, cycle_elapsed_s: float, pico_age_s: float | None) -> None:
        if self._window_start_s is None:
            self._window_start_s = float(loop_start_s)
        self._loop_ms.append(float(cycle_elapsed_s) * 1000.0)
        self._late_ms.append(max(0.0, float(cycle_elapsed_s) - self._target_period_s) * 1000.0)
        self._work_ms.append(float(work_elapsed_s) * 1000.0)
        if pico_age_s is not None:
            self._pico_age_ms.append(float(pico_age_s) * 1000.0)
        if cycle_elapsed_s > self._target_period_s + self._deadline_miss_tolerance_s:
            self._deadline_miss_count += 1
        if work_elapsed_s > self._target_period_s + 1e-9:
            self._work_overrun_count += 1
        if loop_start_s - self._window_start_s >= self._log_interval_s:
            self._emit(loop_start_s)

    def _emit(self, end_s: float) -> None:
        sample_count = len(self._loop_ms)
        if sample_count <= 0:
            self._reset(end_s)
            return
        if not self._enabled:
            self._reset(end_s)
            return
        loop_summary = self._summarize(self._loop_ms)
        late_summary = self._summarize(self._late_ms)
        work_summary = self._summarize(self._work_ms)
        message = (
            "Timing stats | samples=%d window=%.1fs | "
            "loop_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "late_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f deadline_miss(>%.2fms)=%d/%d | "
            "work_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f work_overrun=%d/%d"
        )
        args: list[object] = [
            sample_count,
            end_s - float(self._window_start_s),
            *loop_summary,
            *late_summary,
            self._deadline_miss_tolerance_s * 1000.0,
            self._deadline_miss_count,
            sample_count,
            *work_summary,
            self._work_overrun_count,
            sample_count,
        ]
        if self._pico_age_ms:
            message += " | reference_age_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f"
            args.extend(self._summarize(self._pico_age_ms))
        operator_logger.info(message, *args)
        self._reset(end_s)

    def _reset(self, window_start_s: float) -> None:
        self._window_start_s = float(window_start_s)
        self._loop_ms.clear()
        self._late_ms.clear()
        self._work_ms.clear()
        self._pico_age_ms.clear()
        self._deadline_miss_count = 0
        self._work_overrun_count = 0

    @staticmethod
    def _summarize(samples: list[float]) -> tuple[float, float, float, float]:
        values = np.asarray(samples, dtype=np.float64)
        if values.size <= 0:
            return 0.0, 0.0, 0.0, 0.0
        p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
        return float(p50), float(p95), float(p99), float(np.max(values))


def _parse_sim2real_viewers(cfg: Any) -> set[str]:
    viewers = parse_viewers(cfg)
    unsupported = viewers.difference({"retarget"})
    if unsupported:
        raise ValueError(
            f"Sim2real supports only the optional 'retarget' viewer; got unsupported viewers {sorted(unsupported)}. "
            "Use viewers=retarget or viewers=none."
        )
    return viewers


class _Sim2RealRetargetViewer:
    def __init__(self, *, xml_path: str | None, enabled: bool) -> None:
        self._entry: tuple[Any, Any, Any, Any] | None = None
        if not enabled:
            return
        if not xml_path:
            raise ValueError("Sim2real retarget viewer requires robot.xml_path to be set.")
        self._entry = start_robot_viewer(xml_path, FULL_QPOS_DIM, True, "Retarget", 900, 50)

    def write(self, qpos: Float64Array) -> None:
        if self._entry is None:
            return
        _, arr, alive, _ = self._entry
        if not alive.value:
            return
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < FULL_QPOS_DIM:
            return
        with arr.get_lock():
            arr[:FULL_QPOS_DIM] = qpos[:FULL_QPOS_DIM].tolist()

    def shutdown(self) -> None:
        if self._entry is None:
            return
        proc, _, _, shutdown = self._entry
        shutdown.set()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
        self._entry = None


def _plain_cfg(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    raise TypeError(f"Unsupported sim2real cfg type for multiprocessing: {type(cfg)!r}")


def _mp_cfg(cfg: Any) -> Any:
    return cfg_get(cfg, "runtime", {}) or {}


def _input_provider_kind(cfg: Any) -> str:
    return str(cfg_get(cfg_get(cfg, "input", {}) or {}, "provider", "bvh")).strip().lower()


def _recording_cfg(cfg: Any) -> Any:
    return cfg_get(cfg, "recording", {}) or {}


def _recording_enabled(cfg: Any) -> bool:
    return bool(cfg_get(_recording_cfg(cfg), "enabled", False))


def _recording_camera_cfg(cfg: Any) -> Any:
    return cfg_get(_recording_cfg(cfg), "camera", {}) or {}


def _configured_open_hand_pose(cfg: Any) -> tuple[np.ndarray, np.ndarray]:
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    driver = str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
    if bool(cfg_get(hands_cfg, "enabled", False)):
        if driver == "linkerhand_o6":
            hand_cfg = parse_linkerhand_o6_config(cfg)
        else:
            hand_cfg = parse_linkerhand_l6_config(cfg)
        pose = np.asarray(hand_cfg.open_pose, dtype=np.float32).reshape(-1)
    elif driver == "linkerhand_o6":
        pose = np.array([250, 250, 250, 250, 250, 250], dtype=np.float32)
    else:
        driver_cfg = cfg_get(hands_cfg, "linkerhand_l6", {}) or {}
        thumb_yaw = int(cfg_get(driver_cfg, "thumb_yaw_center", 10))
        pose = np.array([250, thumb_yaw, 250, 250, 250, 250], dtype=np.float32)
    if pose.shape[0] != 6:
        raise ValueError(f"hands.{driver}.open_pose must contain 6 values")
    return pose.copy(), pose.copy()


def _validate_new_runtime_config(cfg: Any) -> None:
    legacy_keys = [key for key in ("sim2real_runtime", "multiprocess", "dexterous_hand") if cfg_get(cfg, key, None) is not None]
    if legacy_keys:
        raise ValueError(
            "Legacy sim2real config keys are no longer supported: "
            f"{', '.join(legacy_keys)}. Use input.provider, runtime, and hands instead."
        )
    provider = _input_provider_kind(cfg)
    if provider not in ("pico4", "bvh"):
        raise ValueError(f"sim2real input.provider must be pico4 or bvh, got {provider!r}")
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    if bool(cfg_get(hands_cfg, "enabled", False)) and provider != "pico4":
        raise ValueError("hands.enabled=true requires input.provider=pico4")
    if _recording_enabled(cfg):
        if provider != "pico4":
            raise ValueError("recording.enabled=true requires input.provider=pico4")
        rec_cfg = _recording_cfg(cfg)
        if str(cfg_get(rec_cfg, "format", "hdf5")) != "hdf5":
            raise ValueError("Only recording.format=hdf5 is supported")
        if str(cfg_get(rec_cfg, "control", "terminal")) != "terminal":
            raise ValueError("Only recording.control=terminal is supported")
        camera_cfg = _recording_camera_cfg(cfg)
        if not bool(cfg_get(camera_cfg, "enabled", True)):
            raise ValueError("recording.camera.enabled=false is not supported for HDF5 recording")
        if str(cfg_get(camera_cfg, "source", "realsense")).lower() != "realsense":
            raise ValueError("recording.camera.source must be realsense")
        if int(cfg_get(rec_cfg, "fps", 30)) != int(cfg_get(camera_cfg, "fps", 30)):
            raise ValueError("recording.fps must match recording.camera.fps")
        input_video = parse_pico_video_config(cfg_get(cfg, "input", {}) or {})
        if not input_video.enabled:
            raise ValueError("recording.enabled=true requires input.video.enabled=true")
        if input_video.source != "realsense":
            raise ValueError("recording.enabled=true requires input.video.source=realsense")
        if int(input_video.width) != int(cfg_get(camera_cfg, "width", 640)):
            raise ValueError("recording.camera.width must match input.video.width")
        if int(input_video.height) != int(cfg_get(camera_cfg, "height", 480)):
            raise ValueError("recording.camera.height must match input.video.height")
        if int(input_video.fps) != int(cfg_get(camera_cfg, "fps", 30)):
            raise ValueError("recording.camera.fps must match input.video.fps")
        input_device = input_video.device
        camera_device = cfg_get(camera_cfg, "device", None)
        camera_device = None if camera_device in (None, "", "null") else str(camera_device)
        if input_device != camera_device:
            raise ValueError("recording.camera.device must match input.video.device")


def _require_recording_dependencies() -> None:
    try:
        from teleopit.recording.hdf5 import TeleopitHDF5Recorder

        TeleopitHDF5Recorder.create
    except Exception as exc:
        raise RuntimeError("HDF5 recording adapter is unavailable") from exc


def _worker_loop(name: str, cfg: dict[str, Any], fn: Callable[[], None]) -> None:
    configure_runtime_logging(cfg, force=True)
    try:
        fn()
    except KeyboardInterrupt:
        pass
    except BaseException:
        logger.exception("%s worker crashed", name)
        raise


def _human_frame_is_valid(frame: object) -> bool:
    return validate_human_frame(frame).valid


class Sim2RealRuntime:
    """Supervisor facade for the process-isolated sim2real runtime."""

    def __init__(self, cfg: Any, *, console: PlainConsole | None = None) -> None:
        self.cfg = _plain_cfg(cfg)
        _validate_new_runtime_config(self.cfg)

        mp_cfg = _mp_cfg(self.cfg)
        video_cfg = parse_pico_video_config(cfg_get(self.cfg, "input", {}))
        if video_cfg.enabled and video_cfg.source not in ("realsense", "test-pattern"):
            raise ValueError(
                "Sim2RealRuntime only supports input.video.source=realsense or test-pattern"
            )
        self._ctx = mp.get_context(str(cfg_get(mp_cfg, "start_method", "spawn")))
        self._stop_event = self._ctx.Event()
        self._processes: list[mp.Process] = []
        self._shutdown_timeout_s = float(cfg_get(mp_cfg, "shutdown_timeout_s", 3.0))
        self._endpoints = default_endpoints(
            host=str(cfg_get(mp_cfg, "host", "127.0.0.1")),
            base_port=int(cfg_get(mp_cfg, "base_port", 39700)),
        )
        self._command_pub: ZmqPublisher | None = None
        self._keyboard: TerminalKeyboardReader | None = None
        self._console = console or PlainConsole(title="Teleopit sim2real", enabled=False)
        self._console_controls = sim2real_keyboard_controls(self.cfg)
        if _recording_enabled(self.cfg):
            _require_recording_dependencies()
            if not sys.stdin.isatty():
                raise RuntimeError("recording.enabled=true requires an interactive TTY for terminal controls")
            if console is None:
                self._console = PlainConsole(title="Teleopit sim2real")

    def run(self) -> None:
        operator_logger.info("runtime starting")
        try:
            self._start_processes()
            if _recording_enabled(self.cfg):
                self._command_pub = ZmqPublisher(self._endpoints.command_pub)
                self._keyboard = TerminalKeyboardReader()
                operator_logger.info("keyboard recording controls active: R start, S save, D discard, Q shutdown, H help")
            while not self._stop_event.is_set():
                self._poll_terminal_recording_controls()
                time.sleep(0.2)
                critical_names = {"robot_control", "reference"}
                if _input_provider_kind(self.cfg) == "pico4":
                    critical_names.add("pico_input")
                critical_dead = [
                    process.name
                    for process in self._processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                    and process.name in critical_names
                ]
                if critical_dead:
                    operator_logger.error("critical worker exited: %s", ", ".join(critical_dead))
                    self._stop_event.set()
                    break
                noncritical_dead = [
                    process.name
                    for process in self._processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                    and process.name not in critical_names
                ]
                if noncritical_dead:
                    operator_logger.warning("non-critical worker exited: %s", ", ".join(noncritical_dead))
        except KeyboardInterrupt:
            operator_logger.info("keyboard interrupt -> shutting down")
            self._stop_event.set()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._command_pub is not None:
            self._command_pub.publish(COMMAND_TOPIC, CommandPacket(command="shutdown", timestamp_s=time.monotonic()))
        for process in self._processes:
            process.join(timeout=self._shutdown_timeout_s)
        for process in self._processes:
            if process.is_alive():
                operator_logger.warning("terminating worker %s", process.name)
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()
        if self._keyboard is not None:
            self._keyboard.close()
            self._keyboard = None
        if self._command_pub is not None:
            self._command_pub.close()
            self._command_pub = None

    def _start_processes(self) -> None:
        if self._processes:
            return

        specs: list[tuple[str, Callable[..., None]]] = []
        if _input_provider_kind(self.cfg) == "pico4":
            specs.append(("pico_input", _run_pico_io_worker))
        specs.extend(
            [
                ("reference", _run_reference_worker),
                ("robot_control", _run_robot_control_worker),
            ]
        )
        hands_cfg = cfg_get(self.cfg, "hands", {}) or {}
        if bool(cfg_get(hands_cfg, "enabled", False)):
            specs.append(("hand_worker", _run_hand_worker))
        if _recording_enabled(self.cfg):
            specs.append(("recording_worker", _run_recording_worker))
        video_cfg = parse_pico_video_config(cfg_get(self.cfg, "input", {}))
        if video_cfg.enabled:
            logger.info("Pico video runs inside pico_input so frames are pushed directly to PicoBridge")

        for name, target in specs:
            process = self._ctx.Process(
                name=name,
                target=target,
                args=(self.cfg, self._endpoints, self._stop_event),
            )
            process.start()
            self._processes.append(process)

    def _poll_terminal_recording_controls(self) -> None:
        if self._keyboard is None or self._command_pub is None:
            return
        events = self._keyboard.poll()
        if not events:
            return
        for event in events:
            normalized = str(event.key).strip().lower()
            if normalized == "h":
                self._console.help(self._console_controls)
                continue
            command = map_recording_key_to_command(event.key)
            if command is None:
                continue
            self._command_pub.publish(COMMAND_TOPIC, CommandPacket(command=command, timestamp_s=time.monotonic()))
            self._console.key_feedback(str(event.key).upper(), _recording_command_label(command))
            if command == "shutdown":
                self._stop_event.set()


def map_recording_key_to_command(key: str) -> str | None:
    normalized = str(key).strip().lower()
    if normalized == "r":
        return "record_start"
    if normalized == "s":
        return "record_save"
    if normalized == "d":
        return "record_discard"
    if normalized == "q":
        return "shutdown"
    return None


def _recording_command_label(command: str) -> str:
    if command == "record_start":
        return "start recording"
    if command == "record_save":
        return "save recording"
    if command == "record_discard":
        return "discard recording"
    if command == "shutdown":
        return "shutdown"
    return command


def _run_pico_io_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        video_cfg = parse_pico_video_config(input_cfg)
        provider = Pico4InputProvider(
            human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
            timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
            buffer_size=int(cfg_get(input_cfg, "pico4_buffer_size", 60)),
            timestamp_gap_reset_s=float(cfg_get(input_cfg, "pico4_timestamp_gap_reset_s", 0.15)),
            pause_button=cfg_get(input_cfg, "pause_button", "A"),
            pause_debounce_s=float(cfg_get(input_cfg, "pause_debounce_s", 0.25)),
            arms_button=cfg_get(input_cfg, "arms_button", "B"),
            arms_debounce_s=float(cfg_get(input_cfg, "arms_debounce_s", cfg_get(input_cfg, "pause_debounce_s", 0.25))),
            bridge_host=str(cfg_get(input_cfg, "bridge_host", "0.0.0.0")),
            bridge_port=int(cfg_get(input_cfg, "bridge_port", 63901)),
            bridge_discovery=bool(cfg_get(input_cfg, "bridge_discovery", True)),
            bridge_advertise_ip=cfg_get(input_cfg, "bridge_advertise_ip", None),
            bridge_video=bridge_video_source(video_cfg),
            bridge_video_enabled=video_cfg.enabled,
            bridge_start_timeout=float(cfg_get(input_cfg, "bridge_start_timeout", 10.0)),
            bridge_history_size=int(cfg_get(input_cfg, "bridge_history_size", 120)),
        )

        body_pub = ZmqPublisher(endpoints.body_pub)
        hand_pub = ZmqPublisher(endpoints.hand_pub)
        controller_pub = ZmqPublisher(endpoints.controller_pub)
        events_pub = ZmqPublisher(endpoints.control_events_pub)
        health_pub = ZmqPublisher(endpoints.health_pub)
        video_pub = ZmqPublisher(endpoints.video_pub) if _recording_enabled(cfg) else None
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        frame_writer: SharedFrameRingWriter | None = None

        def _publish_recording_frame(frame: NDArray[np.generic], timestamp_s: float) -> None:
            nonlocal frame_writer
            if video_pub is None:
                return
            if frame_writer is None:
                frame_writer = SharedFrameRingWriter(
                    shape=tuple(np.asarray(frame).shape),
                    dtype=np.uint8,
                    slots=int(cfg_get(_mp_cfg(cfg), "video_slots", 3)),
                )
            descriptor = frame_writer.write(np.asarray(frame, dtype=np.uint8), timestamp_s=float(timestamp_s))
            video_pub.publish(VIDEO_TOPIC, descriptor)

        video_runtime = PicoVideoRuntime(
            provider=provider,
            config=video_cfg,
            mode="sim2real",
            frame_callback=_publish_recording_frame if _recording_enabled(cfg) else None,
        )

        hz = float(cfg_get(_mp_cfg(cfg), "pico_input_hz", 120.0))
        sleep_s = 1.0 / max(hz, 1.0)
        last_body_seq = -1
        last_hand_seq = -1
        last_controller_seq = -1
        last_video_seq = -1
        last_health_s = 0.0
        try:
            video_runtime.start()
            while not stop_event.is_set():
                video_runtime.tick()
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break

                now = time.monotonic()
                if callable(getattr(provider, "has_frame", None)) and provider.has_frame():
                    try:
                        frame, timestamp_s, seq = provider.get_frame_packet()
                    except Exception:
                        logger.exception("pico_input failed to read body frame")
                    else:
                        if int(seq) != last_body_seq:
                            body_pub.publish(
                                BODY_TOPIC,
                                BodyFramePacket(frame=frame, timestamp_s=float(timestamp_s), seq=int(seq)),
                            )
                            last_body_seq = int(seq)

                events = provider.pop_control_events()
                if events:
                    events_pub.publish(
                        CONTROL_EVENTS_TOPIC,
                        ControlEventsPacket(events=tuple(events), timestamp_s=now, seq=last_body_seq),
                    )

                controller_snapshot = provider.get_controller_snapshot()
                if controller_snapshot is not None and int(controller_snapshot.seq) != last_controller_seq:
                    controller_pub.publish(
                        CONTROLLER_TOPIC,
                        SnapshotPacket(
                            snapshot=controller_snapshot,
                            timestamp_s=float(controller_snapshot.timestamp_s),
                            seq=int(controller_snapshot.seq),
                        ),
                    )
                    last_controller_seq = int(controller_snapshot.seq)

                hand_snapshot = provider.get_hand_snapshot()
                if hand_snapshot is not None and int(hand_snapshot.seq) != last_hand_seq:
                    hand_pub.publish(
                        HAND_TOPIC,
                        SnapshotPacket(
                            snapshot=hand_snapshot,
                            timestamp_s=float(hand_snapshot.timestamp_s),
                            seq=int(hand_snapshot.seq),
                        ),
                    )
                    last_hand_seq = int(hand_snapshot.seq)

                if video_cfg.enabled:
                    last_video_seq = int(video_runtime.pushed_frames)

                if now - last_health_s >= 1.0:
                    health_pub.publish(
                        HEALTH_TOPIC,
                        HealthPacket(
                            worker="pico_input",
                            timestamp_s=now,
                            metrics={
                                "body_seq": last_body_seq,
                                "body_fps": float(provider.fps),
                                "hand_seq": last_hand_seq,
                                "controller_seq": last_controller_seq,
                                "video_seq": last_video_seq,
                            },
                        ),
                    )
                    last_health_s = now
                time.sleep(sleep_s)
        finally:
            video_runtime.stop()
            if frame_writer is not None:
                frame_writer.close(unlink=True)
            command_sub.close()
            for publisher in (body_pub, hand_pub, controller_pub, events_pub, health_pub, video_pub):
                if publisher is not None:
                    publisher.close()
            provider.close()

    _worker_loop("pico_input", cfg, _main)


def _run_reference_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    if _input_provider_kind(cfg) == "bvh":
        _run_bvh_reference_worker(cfg, endpoints, stop_event)
        return
    _run_pico_reference_worker(cfg, endpoints, stop_event)


def _run_pico_reference_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        ref_cfg = parse_reference_config(cfg, provider_fps=None)
        reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / policy_hz,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        if ref_cfg.retarget_buffer_enabled and ref_cfg.reference_delay_s is not None:
            reference_window_builder.validate_runtime_support(
                delay_s=float(ref_cfg.reference_delay_s or 0.0),
                window_s=ref_cfg.retarget_buffer_window_s,
                config_label="Multiprocess sim2real reference timeline",
            )
        timeline = ReferenceTimeline(window_s=ref_cfg.retarget_buffer_window_s) if ref_cfg.retarget_buffer_enabled else None
        reference_manager = (
            RealtimeReferenceManager(
                reference_window_builder=reference_window_builder,
                warmup_steps=ref_cfg.realtime_buffer_warmup_steps,
            )
            if timeline is not None
            else None
        )

        retargeter = RetargetingModule(
            robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
            actual_human_height=float(cfg_get(input_cfg, "human_height", 1.75)),
        )
        body_sub = LatestSubscriber(endpoints.body_pub, BODY_TOPIC)
        health_sub = LatestSubscriber(endpoints.health_pub, HEALTH_TOPIC)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        ref_pub = ZmqPublisher(endpoints.reference_pub)
        idle_sleep_s = float(cfg_get(_mp_cfg(cfg), "retarget_idle_sleep_s", 0.001))
        last_body_seq = -1
        last_body_timestamp_s: float | None = None
        body_dt_s_ema: float | None = None
        latest_body_fps: float | None = None
        resolved_reference_delay_s = (
            float(ref_cfg.reference_delay_s) if ref_cfg.reference_delay_s is not None else None
        )
        runtime_support_validated = ref_cfg.reference_delay_s is not None or not reference_window_builder.requires_timeline
        last_valid_qpos: Float64Array | None = None

        def _publish_invalid_reference(packet: BodyFramePacket, *, elapsed_s: float) -> None:
            qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
            qpos[3] = 1.0
            if last_valid_qpos is not None:
                qpos = np.asarray(last_valid_qpos, dtype=np.float64).copy()
            ref_pub.publish(
                REFERENCE_TOPIC,
                ReferencePacket(
                    qpos=qpos,
                    timestamp_s=time.monotonic(),
                    seq=int(packet.seq),
                    source_timestamp_s=float(packet.timestamp_s),
                    source_seq=int(packet.seq),
                    frame_valid=False,
                    retarget_elapsed_s=elapsed_s,
                ),
            )

        try:
            while not stop_event.is_set():
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break

                health_packet = health_sub.recv_latest()
                if isinstance(health_packet, HealthPacket) and health_packet.worker == "pico_input":
                    metric_fps = health_packet.metrics.get("body_fps")
                    if isinstance(metric_fps, (int, float)) and float(metric_fps) > 0.0:
                        latest_body_fps = float(metric_fps)

                packet = body_sub.recv_latest()
                if packet is None:
                    time.sleep(idle_sleep_s)
                    continue
                if not isinstance(packet, BodyFramePacket) or int(packet.seq) == last_body_seq:
                    continue
                start_s = time.monotonic()
                frame_valid = _human_frame_is_valid(packet.frame)
                if not frame_valid:
                    last_body_seq = int(packet.seq)
                    last_body_timestamp_s = None
                    body_dt_s_ema = None
                    _publish_invalid_reference(packet, elapsed_s=time.monotonic() - start_s)
                    logger.warning("reference worker dropped invalid body frame seq=%s", packet.seq)
                    continue

                try:
                    retargeted = retargeter.retarget(packet.frame)
                    qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
                    reference_window: ReferenceWindow | None = None
                    if timeline is not None:
                        timeline.append(qpos, float(packet.timestamp_s))
                        if reference_manager is not None:
                            reference_manager.note_realtime_frame()
                        if reference_manager is None or not reference_manager.warmup_done:
                            last_body_timestamp_s = float(packet.timestamp_s)
                            last_body_seq = int(packet.seq)
                            continue
                        if last_body_timestamp_s is not None:
                            dt_s = float(packet.timestamp_s) - float(last_body_timestamp_s)
                            if dt_s > 1e-6:
                                body_dt_s_ema = dt_s if body_dt_s_ema is None else 0.9 * body_dt_s_ema + 0.1 * dt_s
                        last_body_timestamp_s = float(packet.timestamp_s)
                        if resolved_reference_delay_s is None:
                            if latest_body_fps is not None and latest_body_fps > 1e-6:
                                resolved_reference_delay_s = 1.0 / latest_body_fps
                            elif body_dt_s_ema is not None and body_dt_s_ema > 1e-6:
                                resolved_reference_delay_s = float(body_dt_s_ema)
                            elif reference_window_builder.requires_timeline:
                                last_body_seq = int(packet.seq)
                                continue
                            else:
                                resolved_reference_delay_s = 0.0
                        if not runtime_support_validated:
                            reference_window_builder.validate_runtime_support(
                                delay_s=float(resolved_reference_delay_s),
                                window_s=ref_cfg.retarget_buffer_window_s,
                                config_label="Multiprocess sim2real reference timeline",
                            )
                            runtime_support_validated = True
                        reference_window, _diag = reference_manager.sample(
                            timeline,
                            time.monotonic() - float(resolved_reference_delay_s),
                        )
                        qpos = reference_window.current_sample().qpos
                    last_valid_qpos = np.asarray(qpos, dtype=np.float64).copy()
                    ref_pub.publish(
                        REFERENCE_TOPIC,
                        ReferencePacket(
                            qpos=np.asarray(qpos, dtype=np.float64).copy(),
                            timestamp_s=time.monotonic(),
                            seq=int(packet.seq),
                            source_timestamp_s=float(packet.timestamp_s),
                            source_seq=int(packet.seq),
                            frame_valid=True,
                            reference_window=reference_window,
                            retarget_elapsed_s=time.monotonic() - start_s,
                        ),
                    )
                    last_body_seq = int(packet.seq)
                except Exception:
                    logger.exception("reference worker failed to retarget body seq=%s", getattr(packet, "seq", None))
        finally:
            body_sub.close()
            health_sub.close()
            command_sub.close()
            ref_pub.close()

    _worker_loop("reference", cfg, _main)


def _run_bvh_reference_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        provider = BVHInputProvider(
            str(cfg_get(input_cfg, "bvh_file", "")),
            human_format=str(cfg_get(input_cfg, "bvh_format", cfg_get(input_cfg, "human_format", "lafan1"))),
        )
        retargeter = RetargetingModule(
            robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(cfg_get(input_cfg, "human_format", cfg_get(input_cfg, "bvh_format", "lafan1"))),
            actual_human_height=float(cfg_get(input_cfg, "human_height", provider.human_height)),
        )
        offline_reference = OfflineReferenceMotion(provider, retargeter)
        playback_cfg = cfg_get(cfg, "playback", {}) or {}
        playback = OfflinePlaybackController(
            duration_s=offline_reference.duration_s,
            step_dt_s=1.0 / policy_hz,
            pause_on_end=bool(cfg_get(playback_cfg, "pause_on_end", True)),
        )
        reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=1.0 / policy_hz,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        ref_pub = ZmqPublisher(endpoints.reference_pub)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        reference_command_sub = LatestSubscriber(endpoints.reference_command_pub, COMMAND_TOPIC)
        mode_sub = LatestSubscriber(endpoints.mode_pub, MODE_TOPIC)
        health_pub = ZmqPublisher(endpoints.health_pub)
        tick_s = 1.0 / policy_hz
        seq = 0
        last_health_s = 0.0
        mocap_active = False

        def _publish(sample_time_s: float, *, frame_valid: bool = True) -> Float64Array | None:
            nonlocal seq
            start_s = time.monotonic()
            sampled = offline_reference.sample(sample_time_s)
            if sampled is None:
                return None
            reference_window = None
            if reference_window_builder.requires_timeline:
                reference_window = build_offline_reference_window(
                    offline_reference,
                    sample_time_s,
                    reference_window_builder,
                    policy_hz,
                )
            qpos = np.asarray(sampled.qpos, dtype=np.float64).copy()
            ref_pub.publish(
                REFERENCE_TOPIC,
                ReferencePacket(
                    qpos=qpos,
                    timestamp_s=time.monotonic(),
                    seq=seq,
                    source_timestamp_s=float(sample_time_s),
                    source_seq=int(sampled.frame_idx0),
                    frame_valid=frame_valid,
                    reference_window=reference_window,
                    retarget_elapsed_s=time.monotonic() - start_s,
                    playback_paused=playback.paused,
                    playback_finished=playback.finished,
                ),
            )
            seq += 1
            return qpos

        try:
            while not stop_event.is_set():
                t0 = time.monotonic()
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket):
                    if command.command == "shutdown":
                        stop_event.set()
                        break
                reference_command = reference_command_sub.recv_latest()
                if isinstance(reference_command, CommandPacket):
                    command = reference_command
                    if command.command == "pause_mocap":
                        playback.pause()
                    elif command.command == "resume_mocap":
                        if not playback.finished:
                            playback.resume()
                    elif command.command == "replay_mocap":
                        playback.replay()
                mode_packet = mode_sub.recv_latest()
                if isinstance(mode_packet, ModeStatePacket):
                    mocap_active = bool(mode_packet.mocap_active)

                qpos = _publish(playback.current_time_s)
                if qpos is None:
                    playback.finish()
                    _publish(playback.current_time_s)
                elif mocap_active:
                    playback.advance()

                now = time.monotonic()
                if now - last_health_s >= 1.0:
                    health_pub.publish(
                        HEALTH_TOPIC,
                        HealthPacket(
                            worker="reference",
                            timestamp_s=now,
                            metrics={
                                "source": "bvh",
                                "seq": seq,
                                "playback_time_s": float(playback.current_time_s),
                                "paused": int(playback.paused),
                                "finished": int(playback.finished),
                            },
                        ),
                    )
                    last_health_s = now
                elapsed = time.monotonic() - t0
                if elapsed < tick_s:
                    time.sleep(tick_s - elapsed)
        finally:
            command_sub.close()
            reference_command_sub.close()
            mode_sub.close()
            ref_pub.close()
            health_pub.close()

    _worker_loop("reference", cfg, _main)


class _RobotControlWorker:
    def __init__(
        self,
        cfg: dict[str, Any],
        endpoints: Sim2RealIpcEndpoints,
        stop_event: MpEvent,
    ) -> None:
        self.cfg = cfg
        self.endpoints = endpoints
        self.stop_event = stop_event
        self.provider_kind = _input_provider_kind(cfg)
        self.mode = RobotMode.IDLE
        self.policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        self.dt = 1.0 / self.policy_hz

        self.robot = UnitreeG1Robot(cfg_get(cfg, "real_robot"))
        self.remote = UnitreeRemote()
        self.policy, self.obs_builder = self._build_policy_and_obs()

        robot_cfg = cfg_get(cfg, "robot")
        self.default_angles = np.asarray(cfg_get(robot_cfg, "default_angles"), dtype=np.float32)
        default_root_qpos = np.asarray(
            cfg_get(robot_cfg, "mujoco_default_qpos", [0.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(-1)
        self._default_root_pos = np.zeros(3, dtype=np.float64)
        if default_root_qpos.shape[0] >= 3:
            self._default_root_pos[:] = default_root_qpos[:3]
        self.num_actions = int(cfg_get(robot_cfg, "num_actions", NUM_JOINTS))
        self._safety = Sim2RealSafetyManager(cfg, self.robot, self.policy_hz, self.num_actions)
        self._standing_return_ramp_duration = float(cfg_get(cfg, "standing_return_ramp_duration", 0.5))
        self._standing_return_kp_ramp_floor_ratio = float(
            cfg_get(cfg, "standing_return_kp_ramp_floor_ratio", 0.5)
        )
        self._arm_joint_indices = parse_arm_joint_indices(cfg, num_actions=self.num_actions)

        self._ref_cfg = parse_reference_config(cfg, provider_fps=None)
        self._reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=self.dt,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        self._ref_proc = Sim2RealReferenceProcessor(
            obs_builder=self.obs_builder,
            policy=self.policy,
            policy_hz=self.policy_hz,
            num_actions=self.num_actions,
            reference_velocity_smoothing_alpha=self._ref_cfg.reference_velocity_smoothing_alpha,
            reference_anchor_velocity_smoothing_alpha=self._ref_cfg.reference_anchor_velocity_smoothing_alpha,
        )

        self._standing_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        self._standing_qpos[3] = 1.0
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None
        self._last_commanded_motion_qpos: Float64Array | None = None
        self._last_mocap_hold_reason: str | None = None
        self._mocap_reentry_armed = False
        self._mocap_session = MocapSessionManager()

        self._latest_reference: ReferencePacket | None = None
        mp_cfg = _mp_cfg(cfg)
        self._max_reference_age_s = float(cfg_get(mp_cfg, "max_reference_age_s", 0.25))
        self._stale_reference_hold_s = float(cfg_get(mp_cfg, "stale_reference_hold_s", 0.08))
        mocap_sw = cfg_get(cfg, "mocap_switch", {}) or {}
        self._check_frames = int(cfg_get(mocap_sw, "check_frames", 10))
        self._last_reference_seq = -1
        self._consecutive_valid_references = 0

        self._reference_sub = LatestSubscriber(endpoints.reference_pub, REFERENCE_TOPIC)
        self._events_sub = LatestSubscriber(endpoints.control_events_pub, CONTROL_EVENTS_TOPIC)
        self._command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        self._reference_command_pub = ZmqPublisher(endpoints.reference_command_pub)
        self._mode_pub = ZmqPublisher(endpoints.mode_pub)
        self._record_pub = ZmqPublisher(endpoints.record_pub) if _recording_enabled(cfg) else None

        viewers = _parse_sim2real_viewers(cfg)
        self._retarget_viewer = _Sim2RealRetargetViewer(
            xml_path=str(cfg_get(robot_cfg, "xml_path", "")) if "retarget" in viewers else None,
            enabled="retarget" in viewers,
        )
        self._mode_seq = 0

    def run(self) -> None:
        operator_logger.info("robot control ready | mode=IDLE | policy_hz=%.0f", self.policy_hz)
        timing = _LoopTimingReporter(
            target_period_s=self.dt,
            log_interval_s=console_timing_interval_s(self.cfg),
            enabled=console_show_timing(self.cfg),
        )
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                self._drain_ipc()

                remote_bytes = self.robot.get_wireless_remote()
                self.remote.update(remote_bytes)
                if self.remote.LB.pressed and self.remote.RB.pressed:
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        operator_logger.warning("DAMPING requested by emergency stop")
                        self._enter_damping()
                else:
                    self._handle_transitions()
                    if self.mode == RobotMode.STANDING:
                        self._standing_step()
                    elif self.mode in (RobotMode.MOCAP, RobotMode.ARMS):
                        self._mocap_step()

                self._publish_mode_state()
                work_elapsed_s = time.monotonic() - t0
                cycle_elapsed_s = self._sleep_until(t0, self.dt)
                timing.record(
                    loop_start_s=t0,
                    work_elapsed_s=work_elapsed_s,
                    cycle_elapsed_s=cycle_elapsed_s,
                    pico_age_s=self._reference_age_s(),
                )
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP, RobotMode.ARMS):
            try:
                self.robot.set_damping()
                time.sleep(0.5)
            except Exception:
                logger.exception("Failed to send damping during robot_control shutdown")
            try:
                self.robot.exit_debug_mode()
            except Exception:
                logger.exception("Failed to exit debug mode during robot_control shutdown")
        self._retarget_viewer.shutdown()
        self._reference_sub.close()
        self._events_sub.close()
        self._command_sub.close()
        self._reference_command_pub.close()
        self._mode_pub.close()
        if self._record_pub is not None:
            self._record_pub.close()
        self.robot.close()

    def _build_policy_and_obs(self) -> tuple[Any, Any]:
        robot_cfg = require_section(self.cfg, "robot")
        controller_cfg = require_section(self.cfg, "controller")
        sim_cfg = build_simulation_cfg(self.cfg)
        policy, obs_builder = _build_policy_components(
            robot_cfg=robot_cfg,
            controller_cfg=controller_cfg,
            sim_cfg=sim_cfg,
            project_root=PROJECT_ROOT,
            controller_cls=RLPolicyController,
        )
        if not bool(getattr(policy, "_multi_input", False)):
            raise ValueError("Sim2real requires an ONNX policy with dual inputs ('obs' and 'obs_history').")
        return policy, obs_builder

    def _drain_ipc(self) -> None:
        command = self._command_sub.recv_latest()
        if isinstance(command, CommandPacket) and command.command == "shutdown":
            self.stop_event.set()
            return
        reference = self._reference_sub.recv_latest()
        if isinstance(reference, ReferencePacket):
            self._note_reference_packet(reference)
        events = self._events_sub.recv_latest()
        if isinstance(events, ControlEventsPacket):
            self._handle_mocap_control_events(events.events)

    def _handle_transitions(self) -> None:
        if self.mode == RobotMode.IDLE:
            if self.remote.start.on_pressed:
                operator_logger.info("Start -> STANDING")
                self._enter_standing()
        elif self.mode == RobotMode.STANDING:
            reentry_request = self._mocap_reentry_armed and self.remote.Y.pressed
            if self.remote.Y.on_pressed or reentry_request:
                if self._can_switch_to_mocap():
                    operator_logger.info("Y -> MOCAP")
                    self._transition_to_mocap()
                else:
                    operator_logger.warning("Y -> waiting for fresh retarget reference")
        elif self.mode in (RobotMode.MOCAP, RobotMode.ARMS):
            if self.provider_kind == "bvh" and self.remote.B.on_pressed:
                operator_logger.info("B -> replay BVH from frame 0")
                self._send_reference_command("replay_mocap")
                self._resume_paused_mocap_if_needed()
                return
            if self.remote.A.on_pressed:
                if self._mocap_session.state == MocapSessionState.PAUSED:
                    operator_logger.info("A -> resume playback")
                    self._send_reference_command("resume_mocap")
                    self._resume_paused_mocap()
                else:
                    operator_logger.info("A -> pause playback")
                    self._send_reference_command("pause_mocap")
                    self._pause_active_mocap()
                return
            if self.remote.X.on_pressed:
                operator_logger.info("X -> STANDING")
                self._enter_standing()
        elif self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                operator_logger.info("Start -> STANDING")
                self._enter_standing()

    def _standing_step(self) -> None:
        robot_state = self.robot.get_state()
        qpos = self._standing_qpos.copy()
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
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._publish_record_step(robot_state=robot_state, reference_qpos=qpos)
        self._write_retarget_viewer(qpos)

    def _mocap_step(self) -> None:
        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._paused_mocap_step()
            return

        reference = self._latest_reference
        age_s = self._reference_age_s()
        if reference is None or age_s is None:
            self._hold_mocap_reference("no retarget reference")
            return
        if not reference.frame_valid:
            self._hold_mocap_reference("invalid retarget reference")
            return
        if age_s > self._stale_reference_hold_s:
            self._hold_mocap_reference(
                "delayed retarget reference",
                detail=f"age={age_s:.3f}s",
            )
            return

        robot_state = self.robot.get_state()
        self._execute_mocap_pipeline(reference.qpos, robot_state, reference.reference_window)

    def _execute_mocap_pipeline(
        self,
        reference_qpos: Float64Array,
        robot_state: object,
        reference_window: ReferenceWindow | None,
    ) -> None:
        reference_window_aligned = False
        reference_qpos = self._ref_proc.align_reference_yaw(reference_qpos, robot_state=robot_state)
        if self.mode == RobotMode.ARMS:
            reference_qpos = self._compose_arm_reference(reference_qpos)
            aligned_window = self._ref_proc.align_reference_window(reference_window, robot_state)
            reference_window = self._compose_arm_reference_window(aligned_window)
            reference_window_aligned = True
        qpos = reference_qpos.copy()
        if qpos.shape[0] < 7 + self.num_actions:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= {7 + self.num_actions})")
        motion_joint_pos = np.asarray(qpos[7:7 + self.num_actions], dtype=np.float32)
        if self._last_retarget_qpos is None:
            raw_motion_joint_vel = np.zeros((self.num_actions,), dtype=np.float32)
        else:
            prev_joint_pos = np.asarray(self._last_retarget_qpos[7:7 + self.num_actions], dtype=np.float32)
            raw_motion_joint_vel = (motion_joint_pos - prev_joint_pos) * np.float32(self.policy_hz)
        motion_joint_vel = self._ref_proc.apply_joint_vel_smoothing(raw_motion_joint_vel)

        anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)
        if not obs_builder_requires_reference_window(self.obs_builder):
            raw_lin, raw_ang = self._ref_proc.compute_anchor_velocities(reference_qpos)
            anchor_lin_vel_w, anchor_ang_vel_w = self._ref_proc.apply_anchor_vel_smoothing(raw_lin, raw_ang)

        motion_qpos = np.asarray(qpos[:7 + self.num_actions], dtype=np.float32)
        obs = self._ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=anchor_lin_vel_w,
            anchor_ang_vel_w=anchor_ang_vel_w,
            reference_window=reference_window,
            reference_window_aligned=reference_window_aligned,
        )
        obs = self._ref_proc.validate_observation(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = reference_qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._last_mocap_hold_reason = None
        self._publish_record_step(robot_state=robot_state, reference_qpos=qpos)
        self._write_retarget_viewer(qpos)

    def _compose_arm_reference(self, retarget_qpos: Float64Array) -> Float64Array:
        return compose_arm_reference(
            standing_qpos=self._standing_qpos,
            retarget_qpos=retarget_qpos,
            arm_joint_indices=self._arm_joint_indices,
            num_actions=self.num_actions,
        )

    def _compose_arm_reference_window(self, reference_window: ReferenceWindow | None) -> ReferenceWindow | None:
        return compose_arm_reference_window(
            reference_window,
            standing_qpos=self._standing_qpos,
            arm_joint_indices=self._arm_joint_indices,
            num_actions=self.num_actions,
        )

    def _enter_standing(self) -> None:
        prev_mode = self.mode
        already_in_debug = self.mode in (RobotMode.STANDING, RobotMode.MOCAP, RobotMode.ARMS)
        if not already_in_debug:
            logger.info("Entering debug mode...")
            ok = self.robot.enter_debug_mode()
            if not ok:
                logger.error("Failed to enter debug mode -- staying in %s", self.mode.value)
                return
            time.sleep(0.5)

        state = self.robot.get_state()
        if prev_mode not in (RobotMode.MOCAP, RobotMode.ARMS):
            logger.info("Locking joints to current position...")
            self.robot.lock_all_joints()
            time.sleep(0.3)

        init_qpos = self._build_robot_state_qpos(state)
        self._last_retarget_qpos = init_qpos
        self._ref_proc.last_reference_qpos = None
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._set_default_standing_reference(state)
        self._reset_policy_state()
        if prev_mode in (RobotMode.MOCAP, RobotMode.ARMS):
            self._safety.start_kp_ramp(
                duration_s=self._standing_return_ramp_duration,
                floor_ratio=self._standing_return_kp_ramp_floor_ratio,
            )
        else:
            self._safety.start_kp_ramp()
        self._mocap_reentry_armed = prev_mode in (RobotMode.MOCAP, RobotMode.ARMS)
        self.mode = RobotMode.STANDING
        operator_logger.info("mode -> STANDING")

    def _can_switch_to_mocap(self) -> bool:
        age_s = self._reference_age_s()
        if self._latest_reference is None or age_s is None:
            return False
        if not self._latest_reference.frame_valid:
            return False
        if self.provider_kind == "bvh":
            return True
        if age_s > self._max_reference_age_s:
            return False
        if self._consecutive_valid_references < self._check_frames:
            logger.warning(
                "Mocap check: only %d/%d valid references",
                self._consecutive_valid_references,
                self._check_frames,
            )
            return False
        return True

    def _transition_to_mocap(self) -> None:
        state = self.robot.get_state()
        last_commanded = getattr(self, "_last_commanded_motion_qpos", None)
        hold_qpos = last_commanded if last_commanded is not None else self._standing_qpos
        resume_qpos = self._build_resume_alignment_qpos(hold_qpos, state)
        self._mocap_reentry_armed = False
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        if self.provider_kind == "bvh":
            self._send_reference_command("replay_mocap")
        self.mode = RobotMode.MOCAP
        operator_logger.info("mode -> MOCAP")

    def _toggle_arms_mode(self) -> None:
        if self.provider_kind != "pico4" or self.mode not in (RobotMode.MOCAP, RobotMode.ARMS):
            return
        if self._mocap_session.state == MocapSessionState.PAUSED:
            logger.info("Ignoring Pico B mode toggle while mocap session is paused")
            return

        state = self.robot.get_state()
        resume_qpos = self._build_resume_alignment_qpos(self._last_commanded_motion_qpos, state)
        next_mode = RobotMode.ARMS if self.mode == RobotMode.MOCAP else RobotMode.MOCAP
        if next_mode == RobotMode.ARMS:
            self._set_default_standing_reference(state)
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        self._safety.start_kp_ramp(
            duration_s=self._standing_return_ramp_duration,
            floor_ratio=self._standing_return_kp_ramp_floor_ratio,
        )
        self.mode = next_mode
        operator_logger.info("mode -> %s", next_mode.value.upper())

    def _resume_paused_mocap_if_needed(self) -> None:
        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._resume_paused_mocap()

    def _enter_damping(self) -> None:
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP, RobotMode.ARMS):
            logger.info("DAMPING: sending LowCmd damping...")
            self.robot.set_damping()
            time.sleep(0.5)
            logger.info("DAMPING: exiting debug mode...")
        self.robot.exit_debug_mode()
        self.mode = RobotMode.DAMPING
        self._publish_damping_record_step()
        self._ref_proc.last_reference_qpos = None
        self._mocap_reentry_armed = False
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._last_mocap_hold_reason = None
        operator_logger.warning("mode -> DAMPING")

    def _reset_policy_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._ref_proc.reset_smoothers()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._last_mocap_hold_reason = None
        self.policy.reset()
        self.obs_builder.reset()

    def _reset_policy_reference_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._ref_proc.reset_smoothers()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self._last_mocap_hold_reason = None
        self.policy.reset()
        self.obs_builder.reset()

    def _build_robot_state_qpos(self, state: object) -> Float64Array:
        qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        qpos[0:3] = self._resolve_base_pos(state)
        qpos[3:7] = np.asarray(getattr(state, "quat"), dtype=np.float64).reshape(-1)[:4]
        qpos[ROOT_DIM:FULL_QPOS_DIM] = np.asarray(getattr(state, "qpos"), dtype=np.float64).reshape(-1)[
            : self.num_actions
        ]
        return qpos

    def _set_default_standing_reference(self, state: object) -> None:
        self._standing_qpos[:] = 0.0
        self._standing_qpos[0:3] = self._resolve_base_pos(state)
        self._standing_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        align_motion_qpos_yaw(np.asarray(getattr(state, "quat"), dtype=np.float32), self._standing_qpos)
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)

    def _resolve_base_pos(self, state: object) -> Float64Array:
        base_pos = getattr(state, "base_pos", None)
        if base_pos is None:
            return self._default_root_pos.copy()
        resolved = self._default_root_pos.copy()
        live = np.asarray(base_pos, dtype=np.float64).reshape(-1)
        resolved[: min(3, live.shape[0])] = live[:3]
        return resolved

    def _build_resume_alignment_qpos(self, hold_qpos: Float64Array | None, state: object) -> Float64Array:
        qpos = self._build_robot_state_qpos(state)
        if hold_qpos is not None:
            qpos[0:2] = np.asarray(hold_qpos, dtype=np.float64).reshape(-1)[0:2]
        base_pos = getattr(state, "base_pos", None)
        if base_pos is not None:
            qpos[0:2] = np.asarray(base_pos, dtype=np.float64).reshape(-1)[0:2]
        return qpos

    def _handle_mocap_control_events(self, control_events: tuple[ControlEvent, ...]) -> None:
        for event in control_events:
            if event.event_type == ControlEventType.TOGGLE_ARMS:
                self._toggle_arms_mode()
                continue
            if event.event_type == ControlEventType.TOGGLE_PAUSE:
                if self.mode not in (RobotMode.MOCAP, RobotMode.ARMS):
                    continue
                if self._mocap_session.state == MocapSessionState.PAUSED:
                    self._resume_paused_mocap()
                else:
                    self._pause_active_mocap()

    def _pause_active_mocap(self) -> None:
        hold_qpos = self._resolve_mocap_hold_qpos()
        self._last_retarget_qpos = hold_qpos.copy()
        self._ref_proc.last_reference_qpos = hold_qpos.copy()
        self._last_commanded_motion_qpos = hold_qpos.copy()
        self._reset_policy_reference_state()
        self._mocap_session.pause(hold_qpos)
        logger.info("Mocap session -> PAUSED (multiprocess episode-reset)")

    def _resume_paused_mocap(self) -> None:
        hold_qpos = self._mocap_session.hold_qpos
        if hold_qpos is None:
            raise RuntimeError("Cannot resume mocap without a paused hold qpos")
        state = self.robot.get_state()
        resume_qpos = self._build_resume_alignment_qpos(hold_qpos, state)
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._reset_policy_reference_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        if self.mode == RobotMode.ARMS:
            self._set_default_standing_reference(state)
            self._safety.start_kp_ramp(
                duration_s=self._standing_return_ramp_duration,
                floor_ratio=self._standing_return_kp_ramp_floor_ratio,
            )
        logger.info("Mocap session -> ACTIVE (multiprocess episode-reset + reference realignment)")

    def _send_reference_command(self, command: str) -> None:
        self._reference_command_pub.publish(
            COMMAND_TOPIC,
            CommandPacket(command=command, timestamp_s=time.monotonic()),
        )

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
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._publish_record_step(robot_state=robot_state, reference_qpos=qpos)
        self._write_retarget_viewer(qpos)

    def _hold_mocap_reference(self, reason: str, *, detail: str | None = None) -> None:
        if self._last_mocap_hold_reason != reason:
            suffix = f" ({detail})" if detail else ""
            logger.warning("Mocap reference not fresh: %s%s -- holding command", reason, suffix)
            self._last_mocap_hold_reason = reason
        hold_qpos = self._resolve_mocap_hold_qpos()
        self._run_static_mocap_step(hold_qpos)

    def _publish_mode_state(self) -> None:
        self._mode_seq += 1
        mocap_like = self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
        active = mocap_like and self._mocap_session.state == MocapSessionState.ACTIVE
        paused = mocap_like and self._mocap_session.state == MocapSessionState.PAUSED
        self._mode_pub.publish(
            MODE_TOPIC,
            ModeStatePacket(
                mode=self.mode.value,
                mocap_active=active,
                mocap_paused=paused,
                timestamp_s=time.monotonic(),
                seq=self._mode_seq,
            ),
        )

    def _publish_record_step(self, *, robot_state: object, reference_qpos: Float64Array) -> None:
        if self._record_pub is None:
            return
        record_mode = self._recording_mode_label()
        mocap_like = self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
        active = mocap_like and self._mocap_session.state == MocapSessionState.ACTIVE
        recordable = self.mode != RobotMode.DAMPING
        try:
            self._record_pub.publish(
                RECORD_TOPIC,
                RecordStepPacket(
                    timestamp_s=time.monotonic(),
                    mode=record_mode,
                    mocap_active=active,
                    recordable=recordable,
                    observation_state=build_observation_state(robot_state).astype(np.float32, copy=True),
                    observation_mode=build_mode_observation(record_mode).astype(np.float32, copy=True),
                    action_reference_qpos=normalize_action_reference_qpos(reference_qpos).astype(np.float32, copy=True),
                    seq=self._mode_seq,
                ),
            )
        except Exception:
            logger.exception("Failed to publish sim2real recording step")

    def _publish_damping_record_step(self) -> None:
        if self._record_pub is None:
            return
        try:
            robot_state = self.robot.get_state()
            reference_qpos = self._build_robot_state_qpos(robot_state)
            self._record_pub.publish(
                RECORD_TOPIC,
                RecordStepPacket(
                    timestamp_s=time.monotonic(),
                    mode=RobotMode.DAMPING.value,
                    mocap_active=False,
                    recordable=False,
                    observation_state=build_observation_state(robot_state).astype(np.float32, copy=True),
                    observation_mode=np.array([-1.0], dtype=np.float32),
                    action_reference_qpos=normalize_action_reference_qpos(reference_qpos).astype(np.float32, copy=True),
                    seq=self._mode_seq,
                ),
            )
        except Exception:
            logger.exception("Failed to publish sim2real damping recording state")

    def _recording_mode_label(self) -> str:
        if (
            self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
            and self._mocap_session.state == MocapSessionState.PAUSED
        ):
            return "pause"
        return self.mode.value

    def _write_retarget_viewer(self, qpos: Float64Array) -> None:
        try:
            self._retarget_viewer.write(qpos)
        except Exception:
            logger.exception("Sim2real retarget viewer update failed; control continues")

    def _reference_age_s(self) -> float | None:
        if self._latest_reference is None:
            return None
        return max(0.0, time.monotonic() - float(self._latest_reference.timestamp_s))

    def _note_reference_packet(self, reference: ReferencePacket) -> None:
        if int(reference.seq) <= self._last_reference_seq:
            return
        self._last_reference_seq = int(reference.seq)
        self._latest_reference = reference
        if (
            self.provider_kind == "bvh"
            and bool(getattr(reference, "playback_paused", False))
            and self.mode in (RobotMode.MOCAP, RobotMode.ARMS)
            and self._mocap_session.state == MocapSessionState.ACTIVE
        ):
            self._pause_active_mocap()
        if not reference.frame_valid:
            self._consecutive_valid_references = 0
            return
        self._consecutive_valid_references += 1

    @staticmethod
    def _sleep_until(t0: float, dt: float) -> float:
        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
        return time.monotonic() - t0


def _run_robot_control_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        worker = _RobotControlWorker(cfg, endpoints, stop_event)
        worker.run()

    _worker_loop("robot_control", cfg, _main)


class _RecordingWorker:
    def __init__(
        self,
        cfg: dict[str, Any],
        endpoints: Sim2RealIpcEndpoints,
        stop_event: MpEvent,
        *,
        recorder_factory: Callable[..., Any] | None = None,
        frame_reader: SharedFrameRingReader | None = None,
    ) -> None:
        self.cfg = cfg
        self.endpoints = endpoints
        self.stop_event = stop_event
        self.rec_cfg = _recording_cfg(cfg)
        self.camera_cfg = _recording_camera_cfg(cfg)
        self.record_modes = {
            str(mode).lower()
            for mode in cfg_get(self.rec_cfg, "record_modes", ["standing", "mocap", "arms", "pause"])
        }
        self.min_episode_seconds = float(cfg_get(self.rec_cfg, "min_episode_seconds", 1.0))
        self.discard_on_shutdown = bool(cfg_get(self.rec_cfg, "discard_on_shutdown", True))
        self.task = str(cfg_get(self.rec_cfg, "task", "demo"))
        self.fps = int(cfg_get(self.rec_cfg, "fps", 30))
        self._record_sub = LatestSubscriber(endpoints.record_pub, RECORD_TOPIC)
        self._video_sub = LatestSubscriber(endpoints.video_pub, VIDEO_TOPIC)
        self._hand_command_sub = LatestSubscriber(endpoints.hand_command_pub, HAND_COMMAND_TOPIC)
        self._command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        self._frame_reader = frame_reader or SharedFrameRingReader()
        self._latest_record: RecordStepPacket | None = None
        left_open, right_open = _configured_open_hand_pose(cfg)
        self._latest_hand_command = HandCommandPacket(
            timestamp_s=0.0,
            driver=str(cfg_get(cfg_get(cfg, "hands", {}) or {}, "driver", "linkerhand_l6")).strip().lower(),
            mode=str(cfg_get(cfg_get(cfg, "hands", {}) or {}, "mode", "gripper")).strip().lower(),
            active=False,
            left_pose=left_open.astype(np.float32, copy=True),
            right_pose=right_open.astype(np.float32, copy=True),
            seq=0,
        )
        self._latest_video_seq = -1
        self._active = False
        self._episode_started_s = 0.0
        self._episode_frames = 0

        from teleopit.recording.hdf5 import (
            TeleopitHDF5Recorder,
            build_recording_schema,
        )

        self._schema = build_recording_schema(self.camera_cfg)
        factory = recorder_factory or TeleopitHDF5Recorder.create
        self._recorder = factory(
            output_dir=cfg_get(self.rec_cfg, "output_dir", "data/recordings/sim2real_hdf5"),
            task=self.task,
            fps=self.fps,
            schema=self._schema,
        )

    def run(self) -> None:
        operator_logger.info("recording worker ready | fps=%d | modes=%s", self.fps, sorted(self.record_modes))
        idle_sleep_s = 1.0 / max(float(self.fps) * 4.0, 1.0)
        try:
            while not self.stop_event.is_set():
                command = self._command_sub.recv_latest()
                if isinstance(command, CommandPacket):
                    if self._handle_command(command):
                        break

                record = self._record_sub.recv_latest()
                if isinstance(record, RecordStepPacket):
                    self._latest_record = record

                hand_command = self._hand_command_sub.recv_latest()
                if isinstance(hand_command, HandCommandPacket):
                    self._latest_hand_command = hand_command

                video = self._video_sub.recv_latest()
                if isinstance(video, SharedFrameDescriptor):
                    self._handle_video(video)

                time.sleep(idle_sleep_s)
        finally:
            if self._active:
                if self.discard_on_shutdown:
                    self._discard_episode("shutdown")
                else:
                    self._save_episode()
            try:
                self._recorder.finalize()
            finally:
                self._record_sub.close()
                self._video_sub.close()
                self._hand_command_sub.close()
                self._command_sub.close()
                self._frame_reader.close()

    def _handle_command(self, command: CommandPacket) -> bool:
        name = command.command
        if name == "shutdown":
            self.stop_event.set()
            return True
        if name == "record_start":
            self._start_episode()
        elif name == "record_save":
            self._save_episode()
        elif name == "record_discard":
            self._discard_episode("manual discard")
        return False

    def _start_episode(self) -> None:
        if self._active:
            logger.warning("Recording episode already active; ignoring R")
            return
        record = self._latest_record
        if record is None:
            logger.warning("Cannot start recording: no robot record packet yet")
            return
        mode = str(record.mode).lower()
        if mode not in self.record_modes or not bool(record.recordable):
            logger.warning(
                "Cannot start recording: mode=%s recordable=%s",
                record.mode,
                record.recordable,
            )
            return
        self._recorder.start_episode()
        self._active = True
        self._episode_started_s = time.monotonic()
        self._episode_frames = 0
        operator_logger.info("recording episode started")

    def _save_episode(self) -> None:
        if not self._active:
            operator_logger.info("no active recording episode to save")
            return
        duration_s = time.monotonic() - self._episode_started_s
        if self._episode_frames <= 0:
            self._discard_episode("empty episode")
            return
        if duration_s < self.min_episode_seconds:
            self._discard_episode(f"short episode ({duration_s:.2f}s < {self.min_episode_seconds:.2f}s)")
            return
        self._recorder.save_episode()
        operator_logger.info("recording episode saved | frames=%d duration=%.2fs", self._episode_frames, duration_s)
        self._active = False
        self._episode_frames = 0

    def _discard_episode(self, reason: str) -> None:
        if not self._active:
            operator_logger.info("no active recording episode to discard")
            return
        self._recorder.discard_episode()
        operator_logger.info("recording episode discarded | reason=%s | frames=%d", reason, self._episode_frames)
        self._active = False
        self._episode_frames = 0

    def _handle_video(self, descriptor: SharedFrameDescriptor) -> None:
        if int(descriptor.seq) == self._latest_video_seq:
            return
        self._latest_video_seq = int(descriptor.seq)
        if not self._active:
            return
        record = self._latest_record
        if record is None:
            return
        mode = str(record.mode).lower()
        if mode not in self.record_modes or not bool(record.recordable):
            logger.warning("Recording stopped because mode is no longer recordable: %s", record.mode)
            self._discard_episode("mode not recordable")
            return
        image = self._frame_reader.read(descriptor, copy=True)
        self._recorder.add_frame(
            image=np.asarray(image, dtype=np.uint8),
            state=np.asarray(record.observation_state, dtype=np.float32),
            mode=np.asarray(record.observation_mode, dtype=np.float32),
            action=np.asarray(record.action_reference_qpos, dtype=np.float32),
            hand_action=normalize_hand_action(
                self._latest_hand_command.left_pose,
                self._latest_hand_command.right_pose,
            ),
            task=self.task,
        )
        self._episode_frames += 1

def _run_recording_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        worker = _RecordingWorker(cfg, endpoints, stop_event)
        worker.run()

    _worker_loop("recording_worker", cfg, _main)


class _HandSnapshotProxy:
    def __init__(self) -> None:
        self.hand_snapshot: Any | None = None
        self.controller_snapshot: Any | None = None

    def get_hand_snapshot(self) -> Any | None:
        return self.hand_snapshot

    def get_controller_snapshot(self) -> Any | None:
        return self.controller_snapshot


def _run_hand_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        proxy = _HandSnapshotProxy()
        runtime = build_hand_runtime(cfg)
        hand_sub = LatestSubscriber(endpoints.hand_pub, HAND_TOPIC)
        controller_sub = LatestSubscriber(endpoints.controller_pub, CONTROLLER_TOPIC)
        mode_sub = LatestSubscriber(endpoints.mode_pub, MODE_TOPIC)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        hand_command_pub = ZmqPublisher(endpoints.hand_command_pub)
        active = False
        hz = float(cfg_get(_mp_cfg(cfg), "hand_worker_hz", 120.0))
        sleep_s = 1.0 / max(hz, 1.0)
        hands_cfg = cfg_get(cfg, "hands", {}) or {}
        driver = str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
        hand_mode = str(cfg_get(hands_cfg, "mode", "gripper")).strip().lower()
        left_pose, right_pose = _configured_open_hand_pose(cfg)
        command_seq = 0

        def _apply_hand_commands(commands: tuple[HandPoseCommand, ...]) -> bool:
            nonlocal left_pose, right_pose
            changed = False
            for hand_command in commands:
                pose = np.asarray(hand_command.pose, dtype=np.float32).reshape(-1)
                if pose.shape[0] != 6:
                    logger.warning("Ignoring %s hand command with invalid pose shape %s", hand_command.side, pose.shape)
                    continue
                if hand_command.side == "left":
                    left_pose = pose.copy()
                    changed = True
                elif hand_command.side == "right":
                    right_pose = pose.copy()
                    changed = True
                else:
                    logger.warning("Ignoring hand command with unsupported side %r", hand_command.side)
            return changed

        def _publish_hand_command(*, timestamp_s: float, active_state: bool) -> None:
            nonlocal command_seq
            command_seq += 1
            hand_command_pub.publish(
                HAND_COMMAND_TOPIC,
                HandCommandPacket(
                    timestamp_s=float(timestamp_s),
                    driver=driver,
                    mode=hand_mode,
                    active=bool(active_state),
                    left_pose=np.asarray(left_pose, dtype=np.float32).copy(),
                    right_pose=np.asarray(right_pose, dtype=np.float32).copy(),
                    seq=command_seq,
                ),
            )

        try:
            startup_commands = runtime.start()
            startup_s = time.monotonic()
            _apply_hand_commands(startup_commands)
            _publish_hand_command(timestamp_s=startup_s, active_state=False)
            while not stop_event.is_set():
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break
                hand_packet = hand_sub.recv_latest()
                if isinstance(hand_packet, SnapshotPacket):
                    proxy.hand_snapshot = hand_packet.snapshot
                controller_packet = controller_sub.recv_latest()
                if isinstance(controller_packet, SnapshotPacket):
                    proxy.controller_snapshot = controller_packet.snapshot
                mode_packet = mode_sub.recv_latest()
                if isinstance(mode_packet, ModeStatePacket):
                    active = bool(mode_packet.mocap_active)
                try:
                    now_s = time.monotonic()
                    commands = runtime.tick(
                        controller_snapshot=proxy.controller_snapshot,
                        hand_snapshot=proxy.hand_snapshot,
                        active=active,
                        now_s=now_s,
                    )
                    if commands:
                        if _apply_hand_commands(commands):
                            _publish_hand_command(timestamp_s=now_s, active_state=active)
                except Exception:
                    logger.exception("Dexterous hand worker tick failed; hand control continues")
                time.sleep(sleep_s)
        finally:
            try:
                shutdown_commands = runtime.close()
                shutdown_s = time.monotonic()
                if _apply_hand_commands(shutdown_commands):
                    _publish_hand_command(timestamp_s=shutdown_s, active_state=False)
            finally:
                hand_sub.close()
                controller_sub.close()
                mode_sub.close()
                command_sub.close()
                hand_command_pub.close()

    _worker_loop("hand_worker", cfg, _main)
