"""Multiprocess sim2real runtime using ZMQ and shared memory."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS, ROOT_DIM
from teleopit.controllers.observation import VelCmdObservationBuilder, align_motion_qpos_yaw
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.pico_video import bridge_video_source, parse_pico_video_config
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.common import cfg_get, require_section
from teleopit.runtime.factory import _build_policy_components, build_simulation_cfg
from teleopit.runtime.mocap_session import MocapSessionManager, MocapSessionState
from teleopit.runtime.reference_config import parse_reference_config
from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindow, ReferenceWindowBuilder
from teleopit.sim.reference_utils import build_static_reference_window, obs_builder_requires_reference_window
from teleopit.sim.realtime_utils import RealtimeReferenceManager
from teleopit.sim2real.controller import (
    RobotMode,
    _LoopTimingReporter,
    _parse_sim2real_viewers,
    _Sim2RealRetargetViewer,
)
from teleopit.sim2real.dexterous_hand import build_linkerhand_runtime
from teleopit.sim2real.mp.ipc import (
    BODY_TOPIC,
    COMMAND_TOPIC,
    CONTROL_EVENTS_TOPIC,
    CONTROLLER_TOPIC,
    HAND_TOPIC,
    HEALTH_TOPIC,
    MODE_TOPIC,
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
    HealthPacket,
    ModeStatePacket,
    ReferencePacket,
    SharedFrameDescriptor,
    SnapshotPacket,
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

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_sim2real_runtime_mode(cfg: Any) -> str:
    """Resolve ``auto|single_process|multiprocess`` into a concrete runtime."""
    raw = str(cfg_get(cfg, "sim2real_runtime", "auto")).strip().lower()
    if raw in ("single", "single_process", "legacy"):
        return "single_process"
    if raw in ("mp", "multi", "multiprocess"):
        provider = str(cfg_get(cfg_get(cfg, "input", {}), "provider", "")).lower()
        if provider != "pico4":
            raise ValueError("sim2real_runtime=multiprocess currently requires input.provider=pico4")
        return "multiprocess"
    if raw != "auto":
        raise ValueError("sim2real_runtime must be auto, single_process, or multiprocess")
    provider = str(cfg_get(cfg_get(cfg, "input", {}), "provider", "")).lower()
    return "multiprocess" if provider == "pico4" else "single_process"


def _plain_cfg(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    raise TypeError(f"Unsupported sim2real cfg type for multiprocessing: {type(cfg)!r}")


def _mp_cfg(cfg: Any) -> Any:
    return cfg_get(cfg, "multiprocess", {}) or {}


def _worker_loop(name: str, fn: Callable[[], None]) -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        fn()
    except KeyboardInterrupt:
        pass
    except BaseException:
        logger.exception("%s worker crashed", name)
        raise


def _human_frame_is_valid(frame: object, *, max_pos_value: float) -> bool:
    if not isinstance(frame, dict):
        return False
    max_pos = float(max_pos_value)
    if not np.isfinite(max_pos) or max_pos <= 0.0:
        return False
    for value in frame.values():
        try:
            pos, quat = value
        except Exception:
            return False
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        quat_arr = np.asarray(quat, dtype=np.float64).reshape(-1)
        if np.any(np.isnan(pos_arr)) or np.any(np.isinf(pos_arr)):
            return False
        if np.any(np.abs(pos_arr) > max_pos):
            return False
        if np.any(np.isnan(quat_arr)) or np.any(np.isinf(quat_arr)):
            return False
    return True


class MultiprocessSim2RealController:
    """Supervisor facade for the multiprocess Pico sim2real runtime."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = _plain_cfg(cfg)
        if resolve_sim2real_runtime_mode(self.cfg) != "multiprocess":
            raise ValueError("MultiprocessSim2RealController requires sim2real_runtime=multiprocess or auto+pico4")

        mp_cfg = _mp_cfg(self.cfg)
        video_cfg = parse_pico_video_config(cfg_get(self.cfg, "input", {}))
        if video_cfg.enabled and video_cfg.source not in ("realsense", "test-pattern"):
            raise ValueError(
                "Multiprocess sim2real only supports input.video.source=realsense or test-pattern"
            )
        self._ctx = mp.get_context(str(cfg_get(mp_cfg, "start_method", "spawn")))
        self._stop_event = self._ctx.Event()
        self._processes: list[mp.Process] = []
        self._shutdown_timeout_s = float(cfg_get(mp_cfg, "shutdown_timeout_s", 3.0))
        self._endpoints = default_endpoints(
            host=str(cfg_get(mp_cfg, "host", "127.0.0.1")),
            base_port=int(cfg_get(mp_cfg, "base_port", 39700)),
        )

    def run(self) -> None:
        logger.info("Starting multiprocess sim2real runtime")
        try:
            self._start_processes()
            while not self._stop_event.is_set():
                time.sleep(0.2)
                critical_dead = [
                    process.name
                    for process in self._processes
                    if not process.is_alive()
                    and process.exitcode not in (None, 0)
                    and process.name in {"robot_control", "pico_io", "retarget_worker"}
                ]
                if critical_dead:
                    logger.error("Critical sim2real worker exited: %s", ", ".join(critical_dead))
                    self._stop_event.set()
                    break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt -- shutting down multiprocess sim2real")
            self._stop_event.set()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop_event.set()
        for process in self._processes:
            process.join(timeout=self._shutdown_timeout_s)
        for process in self._processes:
            if process.is_alive():
                logger.warning("Terminating sim2real worker %s", process.name)
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()

    def _start_processes(self) -> None:
        if self._processes:
            return

        specs: list[tuple[str, Callable[..., None]]] = [
            ("pico_io", _run_pico_io_worker),
            ("retarget_worker", _run_retarget_worker),
            ("robot_control", _run_robot_control_worker),
        ]
        hand_mode = str(cfg_get(cfg_get(self.cfg, "dexterous_hand", {}) or {}, "mode", "off")).lower()
        if hand_mode != "off":
            specs.append(("hand_worker", _run_hand_worker))
        video_cfg = parse_pico_video_config(cfg_get(self.cfg, "input", {}))
        if video_cfg.enabled and video_cfg.source not in (None, "test-pattern"):
            specs.append(("video_worker", _run_video_worker))
        elif video_cfg.enabled and video_cfg.source == "test-pattern":
            # pico-bridge can generate test-pattern internally without a camera worker.
            logger.info("Pico video test-pattern uses pico_bridge internal source")

        for name, target in specs:
            process = self._ctx.Process(
                name=name,
                target=target,
                args=(self.cfg, self._endpoints, self._stop_event),
            )
            process.start()
            self._processes.append(process)


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
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        video_sub = (
            LatestSubscriber(endpoints.video_pub, VIDEO_TOPIC)
            if video_cfg.enabled and video_cfg.source not in (None, "test-pattern")
            else None
        )
        frame_reader = SharedFrameRingReader()

        hz = float(cfg_get(_mp_cfg(cfg), "pico_io_hz", 120.0))
        sleep_s = 1.0 / max(hz, 1.0)
        last_body_seq = -1
        last_hand_seq = -1
        last_controller_seq = -1
        last_video_seq = -1
        last_health_s = 0.0
        try:
            while not stop_event.is_set():
                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break

                now = time.monotonic()
                if callable(getattr(provider, "has_frame", None)) and provider.has_frame():
                    try:
                        frame, timestamp_s, seq = provider.get_frame_packet()
                    except Exception:
                        logger.exception("pico_io failed to read body frame")
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

                if video_sub is not None:
                    descriptor = video_sub.recv_latest()
                    if isinstance(descriptor, SharedFrameDescriptor):
                        try:
                            frame = frame_reader.read(descriptor, copy=False)
                            provider.push_video_frame(np.asarray(frame, dtype=np.uint8))
                            last_video_seq = int(descriptor.seq)
                        except Exception as exc:
                            logger.warning("Pico video frame dropped: %s", exc)

                if now - last_health_s >= 1.0:
                    health_pub.publish(
                        HEALTH_TOPIC,
                        HealthPacket(
                            worker="pico_io",
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
            frame_reader.close()
            if video_sub is not None:
                video_sub.close()
            command_sub.close()
            for publisher in (body_pub, hand_pub, controller_pub, events_pub, health_pub):
                publisher.close()
            provider.close()

    _worker_loop("pico_io", _main)


def _run_retarget_worker(
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
        mocap_sw = cfg_get(cfg, "mocap_switch", {}) or {}
        max_position_value = float(cfg_get(mocap_sw, "max_position_value", 5.0))
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
                health_packet = health_sub.recv_latest()
                if isinstance(health_packet, HealthPacket) and health_packet.worker == "pico_io":
                    metric_fps = health_packet.metrics.get("body_fps")
                    if isinstance(metric_fps, (int, float)) and float(metric_fps) > 0.0:
                        latest_body_fps = float(metric_fps)

                command = command_sub.recv_latest()
                if isinstance(command, CommandPacket) and command.command == "shutdown":
                    stop_event.set()
                    break

                packet = body_sub.recv_latest()
                if packet is None:
                    time.sleep(idle_sleep_s)
                    continue
                if not isinstance(packet, BodyFramePacket) or int(packet.seq) == last_body_seq:
                    continue
                start_s = time.monotonic()
                frame_valid = _human_frame_is_valid(packet.frame, max_pos_value=max_position_value)
                if not frame_valid:
                    last_body_seq = int(packet.seq)
                    last_body_timestamp_s = None
                    body_dt_s_ema = None
                    _publish_invalid_reference(packet, elapsed_s=time.monotonic() - start_s)
                    logger.warning("retarget_worker dropped invalid body frame seq=%s", packet.seq)
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
                    logger.exception("retarget_worker failed to retarget body seq=%s", getattr(packet, "seq", None))
        finally:
            body_sub.close()
            health_sub.close()
            command_sub.close()
            ref_pub.close()

    _worker_loop("retarget_worker", _main)


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
        self.mode = RobotMode.IDLE
        self.policy_hz = float(cfg_get(cfg, "policy_hz", 50.0))
        self.dt = 1.0 / self.policy_hz

        self.robot = UnitreeG1Robot(cfg_get(cfg, "real_robot"))
        self.remote = UnitreeRemote()
        self.policy, self.obs_builder = self._build_policy_and_obs()

        robot_cfg = cfg_get(cfg, "robot")
        self.default_angles = np.asarray(cfg_get(robot_cfg, "default_angles"), dtype=np.float32)
        self.num_actions = int(cfg_get(robot_cfg, "num_actions", NUM_JOINTS))
        self._safety = Sim2RealSafetyManager(cfg, self.robot, self.policy_hz, self.num_actions)
        self._standing_return_ramp_duration = float(cfg_get(cfg, "standing_return_ramp_duration", 0.5))
        self._standing_return_kp_ramp_floor_ratio = float(
            cfg_get(cfg, "standing_return_kp_ramp_floor_ratio", 0.5)
        )

        self._ref_cfg = parse_reference_config(cfg, provider_fps=None)
        self._reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=self.dt,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )
        mocap_sw = cfg_get(cfg, "mocap_switch", {}) or {}
        self._ref_proc = Sim2RealReferenceProcessor(
            obs_builder=self.obs_builder,
            policy=self.policy,
            policy_hz=self.policy_hz,
            num_actions=self.num_actions,
            reference_velocity_smoothing_alpha=self._ref_cfg.reference_velocity_smoothing_alpha,
            reference_anchor_velocity_smoothing_alpha=self._ref_cfg.reference_anchor_velocity_smoothing_alpha,
            max_pos_value=float(cfg_get(mocap_sw, "max_position_value", 5.0)),
        )

        self._standing_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        self._standing_qpos[3] = 1.0
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None
        self._last_commanded_motion_qpos: Float64Array | None = None
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
        self._mode_pub = ZmqPublisher(endpoints.mode_pub)

        viewers = _parse_sim2real_viewers(cfg)
        self._retarget_viewer = _Sim2RealRetargetViewer(
            xml_path=str(cfg_get(robot_cfg, "xml_path", "")) if "retarget" in viewers else None,
            enabled="retarget" in viewers,
        )
        self._mode_seq = 0

    def run(self) -> None:
        logger.info("Robot control worker started | mode=IDLE | policy_hz=%.0f", self.policy_hz)
        timing = _LoopTimingReporter(target_period_s=self.dt)
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                self._drain_ipc()

                remote_bytes = self.robot.get_wireless_remote()
                self.remote.update(remote_bytes)
                if self.remote.LB.pressed and self.remote.RB.pressed:
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        self._enter_damping()
                else:
                    self._handle_transitions()
                    if self.mode == RobotMode.STANDING:
                        self._standing_step()
                    elif self.mode == RobotMode.MOCAP:
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
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP):
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
        self._mode_pub.close()
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
                logger.info("Start pressed (from IDLE)")
                self._enter_standing()
        elif self.mode == RobotMode.STANDING:
            reentry_request = self._mocap_reentry_armed and self.remote.Y.pressed
            if self.remote.Y.on_pressed or reentry_request:
                if self._can_switch_to_mocap():
                    logger.info("Y pressed -> entering MOCAP")
                    self._transition_to_mocap()
                else:
                    logger.warning("Cannot switch to MOCAP -- no fresh retarget reference")
        elif self.mode == RobotMode.MOCAP:
            if self.remote.A.on_pressed:
                if self._mocap_session.state == MocapSessionState.PAUSED:
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
        self._write_retarget_viewer(qpos)

    def _mocap_step(self) -> None:
        if self._mocap_session.state == MocapSessionState.PAUSED:
            self._paused_mocap_step()
            return

        reference = self._latest_reference
        age_s = self._reference_age_s()
        if reference is None or age_s is None:
            self._hold_or_damp_stale_reference("no retarget reference")
            return
        if not reference.frame_valid:
            logger.warning("Retarget reference invalid -- holding last command")
            self._hold_or_damp_stale_reference("invalid retarget reference")
            return
        if age_s > self._max_reference_age_s:
            logger.warning("Retarget reference stale %.3fs -- entering damping", age_s)
            self._enter_damping()
            return
        if age_s > self._stale_reference_hold_s and self._last_commanded_motion_qpos is not None:
            self._run_static_mocap_step(self._last_commanded_motion_qpos)
            return

        robot_state = self.robot.get_state()
        self._execute_mocap_pipeline(reference.qpos, robot_state, reference.reference_window)

    def _execute_mocap_pipeline(
        self,
        reference_qpos: Float64Array,
        robot_state: object,
        reference_window: ReferenceWindow | None,
    ) -> None:
        reference_qpos = self._ref_proc.align_reference_yaw(reference_qpos, robot_state=robot_state)
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
        )
        obs = self._ref_proc.validate_observation(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self._safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        self._safety.send_positions(target_dof_pos)
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()
        self._ref_proc.last_reference_qpos = reference_qpos.copy()
        self._last_commanded_motion_qpos = qpos.copy()
        self._write_retarget_viewer(qpos)

    def _enter_standing(self) -> None:
        prev_mode = self.mode
        already_in_debug = self.mode in (RobotMode.STANDING, RobotMode.MOCAP)
        if not already_in_debug:
            logger.info("Entering debug mode...")
            ok = self.robot.enter_debug_mode()
            if not ok:
                logger.error("Failed to enter debug mode -- staying in %s", self.mode.value)
                return
            time.sleep(0.5)

        state = self.robot.get_state()
        if prev_mode != RobotMode.MOCAP:
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
        if prev_mode == RobotMode.MOCAP:
            self._safety.start_kp_ramp(
                duration_s=self._standing_return_ramp_duration,
                floor_ratio=self._standing_return_kp_ramp_floor_ratio,
            )
        else:
            self._safety.start_kp_ramp()
        self._mocap_reentry_armed = prev_mode == RobotMode.MOCAP
        self.mode = RobotMode.STANDING
        logger.info("Mode -> STANDING (multiprocess robot control)")

    def _can_switch_to_mocap(self) -> bool:
        age_s = self._reference_age_s()
        if self._latest_reference is None or age_s is None:
            return False
        if not self._latest_reference.frame_valid:
            return False
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
        resume_qpos = self._build_resume_alignment_qpos(self._standing_qpos, state)
        self._mocap_reentry_armed = False
        self._reset_policy_state()
        self._last_retarget_qpos = None
        self._last_commanded_motion_qpos = resume_qpos.copy()
        self._ref_proc.reset_alignment(target_qpos=resume_qpos)
        self.mode = RobotMode.MOCAP
        logger.info("Mode -> MOCAP (tracking multiprocess retarget reference)")

    def _enter_damping(self) -> None:
        if self.mode in (RobotMode.STANDING, RobotMode.MOCAP):
            logger.info("DAMPING: sending LowCmd damping...")
            self.robot.set_damping()
            time.sleep(0.5)
            logger.info("DAMPING: exiting debug mode...")
            self.robot.exit_debug_mode()
        self.mode = RobotMode.DAMPING
        self._ref_proc.last_reference_qpos = None
        self._mocap_reentry_armed = False
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        logger.info("Mode -> DAMPING (press Start to re-enter STANDING)")

    def _reset_policy_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._ref_proc.reset_smoothers()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self.policy.reset()
        self.obs_builder.reset()

    def _reset_policy_reference_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._ref_proc.reset_smoothers()
        self._ref_proc.reset_alignment()
        self._mocap_session.reset()
        self._last_commanded_motion_qpos = None
        self.policy.reset()
        self.obs_builder.reset()

    def _build_robot_state_qpos(self, state: object) -> Float64Array:
        qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        base_pos = getattr(state, "base_pos", None)
        if base_pos is not None:
            qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(-1)[:3]
        qpos[3:7] = np.asarray(getattr(state, "quat"), dtype=np.float64).reshape(-1)[:4]
        qpos[ROOT_DIM:FULL_QPOS_DIM] = np.asarray(getattr(state, "qpos"), dtype=np.float64).reshape(-1)[
            : self.num_actions
        ]
        return qpos

    def _set_default_standing_reference(self, state: object) -> None:
        self._standing_qpos[:] = 0.0
        base_pos = getattr(state, "base_pos", None)
        if base_pos is not None:
            self._standing_qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(-1)[:3]
        self._standing_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        align_motion_qpos_yaw(np.asarray(getattr(state, "quat"), dtype=np.float32), self._standing_qpos)
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)

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
            if event.event_type != ControlEventType.TOGGLE_PAUSE:
                continue
            if self.mode != RobotMode.MOCAP:
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
        logger.info("Mocap session -> ACTIVE (multiprocess episode-reset + reference realignment)")

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
        self._write_retarget_viewer(qpos)

    def _hold_or_damp_stale_reference(self, reason: str) -> None:
        if self._last_commanded_motion_qpos is not None:
            self._run_static_mocap_step(self._last_commanded_motion_qpos)
            return
        logger.warning("No mocap hold pose available after %s -- entering damping", reason)
        self._enter_damping()

    def _publish_mode_state(self) -> None:
        self._mode_seq += 1
        active = self.mode == RobotMode.MOCAP and self._mocap_session.state == MocapSessionState.ACTIVE
        paused = self.mode == RobotMode.MOCAP and self._mocap_session.state == MocapSessionState.PAUSED
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

    _worker_loop("robot_control", _main)


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
        runtime = build_linkerhand_runtime(cfg, proxy)
        hand_sub = LatestSubscriber(endpoints.hand_pub, HAND_TOPIC)
        controller_sub = LatestSubscriber(endpoints.controller_pub, CONTROLLER_TOPIC)
        mode_sub = LatestSubscriber(endpoints.mode_pub, MODE_TOPIC)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        active = False
        hz = float(cfg_get(_mp_cfg(cfg), "hand_worker_hz", 120.0))
        sleep_s = 1.0 / max(hz, 1.0)
        runtime.start()
        try:
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
                    runtime.tick(active=active)
                except Exception:
                    logger.exception("Dexterous hand worker tick failed; hand control continues")
                time.sleep(sleep_s)
        finally:
            try:
                runtime.close()
            finally:
                hand_sub.close()
                controller_sub.close()
                mode_sub.close()
                command_sub.close()

    _worker_loop("hand_worker", _main)


def _run_video_worker(
    cfg: dict[str, Any],
    endpoints: Sim2RealIpcEndpoints,
    stop_event: MpEvent,
) -> None:
    def _main() -> None:
        input_cfg = cfg_get(cfg, "input", {}) or {}
        video_cfg = parse_pico_video_config(input_cfg)
        if not video_cfg.enabled:
            return
        if video_cfg.source not in ("realsense",):
            logger.warning("Multiprocess video worker supports source=realsense; got %s", video_cfg.source)
            return

        writer = SharedFrameRingWriter(
            shape=(video_cfg.height, video_cfg.width, 3),
            dtype=np.uint8,
            slots=int(cfg_get(_mp_cfg(cfg), "video_slots", 3)),
        )
        video_pub = ZmqPublisher(endpoints.video_pub)
        command_sub = LatestSubscriber(endpoints.command_pub, COMMAND_TOPIC)
        try:
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            rs_config = rs.config()
            if video_cfg.device is not None:
                rs_config.enable_device(video_cfg.device)
            rs_config.enable_stream(
                rs.stream.color,
                video_cfg.width,
                video_cfg.height,
                rs.format.rgb8,
                video_cfg.fps,
            )
            pipeline.start(rs_config)
            try:
                while not stop_event.is_set():
                    command = command_sub.recv_latest()
                    if isinstance(command, CommandPacket) and command.command == "shutdown":
                        stop_event.set()
                        break
                    frames = pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    rgb = np.ascontiguousarray(np.asanyarray(color_frame.get_data()), dtype=np.uint8)
                    descriptor = writer.write(rgb, timestamp_s=time.monotonic())
                    video_pub.publish(VIDEO_TOPIC, descriptor)
            finally:
                pipeline.stop()
        finally:
            command_sub.close()
            video_pub.close()
            writer.close(unlink=True)

    _worker_loop("video_worker", _main)
