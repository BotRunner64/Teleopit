"""Pico4 VR full-body motion capture input provider.

Uses the in-process ``pico_bridge`` receiver to collect PICO tracking frames.
The provider converts native PICO/Unity poses (meters, xyzw quaternions) into
Teleopit's realtime ``HumanFrame`` format.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import inspect
import logging
import threading
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import (
    ControlEvent,
    ControlEventType,
    HumanFrame,
    RealtimeInputPacket,
)
from teleopit.inputs.rot_utils import quat_mul_np
from teleopit.interfaces import RealtimeInputProvider
from teleopit.sim.reference_motion import interpolate_human_frames

logger = logging.getLogger(__name__)

# PICO/Unity -> Teleopit retarget input space.
_INPUT_TO_TELEOPIT_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_INPUT_TO_TELEOPIT_QUAT = R.from_matrix(_INPUT_TO_TELEOPIT_MATRIX).as_quat(scalar_first=True)

BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]
BODY_JOINT_PARENTS = np.array(
    [
        -1,
        0, 0, 0, 1, 2,
        3, 4, 5, 6, 7, 8,
        9, 12, 12, 12, 13, 14,
        16, 17, 18, 19, 20, 21,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class PicoControllerState:
    """Latest per-controller input state exposed by pico_bridge."""

    raw: bool
    grip: float
    trigger: float
    present: bool = True


@dataclass(frozen=True)
class PicoControllerSnapshot:
    """Immutable snapshot of Pico controller inputs for auxiliary runtimes."""

    left: PicoControllerState
    right: PicoControllerState
    timestamp_s: float
    seq: int

_PAUSE_BUTTON_MAP: dict[str, tuple[str, str]] = {
    "A": ("right", "primaryButton"),
    "B": ("right", "secondaryButton"),
    "X": ("left", "primaryButton"),
    "Y": ("left", "secondaryButton"),
    "left_axis_click": ("left", "axisClick"),
    "right_axis_click": ("right", "axisClick"),
    "left_menu_button": ("left", "menuButton"),
    "right_menu_button": ("right", "menuButton"),
}


def _bridge_accepts_video_enabled(bridge_cls: type[Any]) -> bool:
    try:
        signature = inspect.signature(bridge_cls)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    if "video_enabled" in parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def _coordinate_transform_input(body_pose_dict: dict[str, list]) -> dict[str, list]:
    """Transform provider-space poses into Teleopit's expected coordinates."""
    for body_name, value in body_pose_dict.items():
        x, y, z = value[0]
        qw, qx, qy, qz = value[1]

        orientation = quat_mul_np(
            _INPUT_TO_TELEOPIT_QUAT, np.array([qw, qx, qy, qz]), scalar_first=True
        )
        position = np.array([x, y, z]) @ _INPUT_TO_TELEOPIT_MATRIX.T

        body_pose_dict[body_name] = [position.tolist(), orientation.tolist()]

    return body_pose_dict


class Pico4InputProvider(RealtimeInputProvider):
    """Realtime input provider backed by the ``pico_bridge`` receiver."""

    def __init__(
        self,
        human_format: str = "pico_bridge",
        timeout: float = 60.0,
        buffer_size: int = 60,
        timestamp_gap_reset_s: float = 0.15,
        pause_button: str | None = "A",
        pause_debounce_s: float = 0.25,
        bridge_host: str = "0.0.0.0",
        bridge_port: int = 63901,
        bridge_discovery: bool = True,
        bridge_advertise_ip: str | None = None,
        bridge_video: str | None = None,
        bridge_video_enabled: bool | None = None,
        bridge_start_timeout: float = 10.0,
        bridge_history_size: int = 120,
        bridge_cls: type[Any] | None = None,
    ) -> None:
        if bridge_cls is None:
            try:
                from pico_bridge import PicoBridge
            except ImportError as exc:
                raise ImportError(
                    "pico_bridge is required for Pico4 input. Install the receiver package, "
                    "for example: pip install -e '.[pico4]'"
                ) from exc
            bridge_cls = PicoBridge
        if not _bridge_accepts_video_enabled(bridge_cls):
            raise RuntimeError(
                "pico_bridge >= 0.2.0 is required for Pico4 input. Reinstall the Pico extra with "
                "pip install -e '.[pico4]' so PicoBridge accepts video_enabled and push_video_frame()."
            )

        self._human_format = human_format
        self._timeout = float(timeout)
        self._closed = False
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._frame_cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size, fps_window=30)
        self._timestamp_gap_reset_s = float(timestamp_gap_reset_s)
        self._pending_control_events: deque[ControlEvent] = deque()
        self._pause_button = None if pause_button in (None, "", "null") else str(pause_button)
        self._pause_debounce_s = max(float(pause_debounce_s), 0.0)
        self._pause_button_path = self._resolve_button_path(self._pause_button)
        self._last_pause_button_pressed = False
        self._last_pause_toggle_timestamp: float | None = None
        self._last_raw_body_joints: NDArray[np.float64] | None = None
        self._last_frame_timestamp: float | None = None
        self._last_source_seq: int | None = None
        self._controller_snapshot: PicoControllerSnapshot | None = None
        self._bridge = bridge_cls(
            host=bridge_host,
            port=int(bridge_port),
            discovery=bool(bridge_discovery),
            advertise_ip=bridge_advertise_ip,
            video=bridge_video,
            video_enabled=bridge_video_enabled,
            history_size=int(bridge_history_size),
            start_timeout=float(bridge_start_timeout),
        )
        self._bridge.start()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="pico4_input")
        self._poll_thread.start()
        if self._pause_button is not None and self._pause_button_path is None:
            logger.warning(
                "Pico4InputProvider pause button '%s' is unsupported by pico_bridge; pause events disabled",
                self._pause_button,
            )
        logger.info("Pico4InputProvider initialized (pico_bridge)")

    @property
    def fps(self) -> float:
        with self._lock:
            return self._frame_cache.fps()

    @property
    def human_format(self) -> str:
        return self._human_format

    @property
    def bone_names(self) -> list[str]:
        return list(BODY_JOINT_NAMES)

    @property
    def bone_parents(self) -> NDArray[np.int32]:
        return BODY_JOINT_PARENTS.copy()

    def is_available(self) -> bool:
        return not self._closed and self._poll_thread.is_alive()

    def get_frame(self) -> HumanFrame:
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No Pico4 body data received within {self._timeout:.1f}s timeout")
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest()

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest_packet()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No Pico4 body data received within {self._timeout:.1f}s timeout")
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest_packet()

    def get_realtime_input_packet(self) -> RealtimeInputPacket[HumanFrame]:
        frame, timestamp_s, seq = self.get_frame_packet()
        with self._lock:
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return RealtimeInputPacket(
            frame=frame,
            timestamp_s=timestamp_s,
            seq=seq,
            control_events=control_events,
        )

    def pop_control_events(self) -> tuple[ControlEvent, ...]:
        with self._lock:
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return control_events

    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        """Return the latest Pico controller-axis snapshot, if one has arrived."""
        with self._lock:
            return self._controller_snapshot

    def push_video_frame(self, frame: NDArray[np.uint8]) -> int:
        """Push one RGB camera frame to pico-bridge 0.2.0 video output."""
        push_video_frame = getattr(self._bridge, "push_video_frame", None)
        if not callable(push_video_frame):
            raise RuntimeError("Installed pico_bridge does not expose push_video_frame(); use pico-bridge 0.2.0")
        return int(push_video_frame(frame))

    def has_frame(self) -> bool:
        with self._lock:
            return len(self._frame_cache) > 0

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(f"No Pico4 body data received within {self._timeout:.1f}s timeout")
        with self._lock:
            buf = self._frame_cache.snapshot()

        if not buf:
            raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
        if len(buf) == 1:
            return buf[0][0]

        target_time = float(query_time_s - max(delay_s, 0.0))
        if target_time <= buf[0][1]:
            return buf[0][0]
        if target_time >= buf[-1][1]:
            return buf[-1][0]

        for i in range(1, len(buf)):
            older_frame, older_ts = buf[i - 1]
            newer_frame, newer_ts = buf[i]
            if target_time <= newer_ts:
                dt = newer_ts - older_ts
                if dt <= 1e-6:
                    return newer_frame
                alpha = float(np.clip((target_time - older_ts) / dt, 0.0, 1.0))
                return interpolate_human_frames(older_frame, newer_frame, alpha)

        return buf[-1][0]

    def close(self) -> None:
        self._closed = True
        self._poll_thread.join(timeout=3.0)
        close = getattr(self._bridge, "close", None)
        if callable(close):
            close()
        logger.info("Pico4InputProvider closed")

    def _poll_loop(self) -> None:
        while not self._closed:
            try:
                frame = self._bridge.wait_frame(timeout=0.1, after_seq=self._last_source_seq)
            except TimeoutError:
                continue
            except Exception:
                if not self._closed:
                    logger.exception("Failed to read pico_bridge frame")
                    time.sleep(0.05)
                continue

            self._accept_pico_frame(frame)

    def _accept_pico_frame(self, frame: Any) -> bool:
        timestamp = float(getattr(frame, "receive_time_s", time.monotonic()))
        self._accept_controller_snapshot(frame, timestamp=timestamp)
        self._poll_control_events(frame, timestamp=timestamp)

        body = getattr(frame, "body", None)
        if body is None or not bool(getattr(body, "active", False)):
            self._last_source_seq = int(getattr(frame, "seq", self._last_source_seq or -1))
            return False

        body_joints = np.asarray(getattr(body, "joints"), dtype=np.float64)
        if body_joints.shape != (len(BODY_JOINT_NAMES), 7):
            logger.warning("Unexpected pico_bridge body joint shape: %s", body_joints.shape)
            self._last_source_seq = int(getattr(frame, "seq", self._last_source_seq or -1))
            return False

        if self._last_raw_body_joints is not None and np.array_equal(body_joints, self._last_raw_body_joints):
            self._last_source_seq = int(getattr(frame, "seq", self._last_source_seq or -1))
            return False

        human_frame = self._convert_body_joints_to_frame(body_joints)
        with self._lock:
            if (
                self._last_frame_timestamp is not None
                and self._timestamp_gap_reset_s > 0.0
                and timestamp - self._last_frame_timestamp > self._timestamp_gap_reset_s
            ):
                self._frame_cache.clear()
                logger.warning(
                    "Pico4InputProvider timestamp-gap reset | gap=%.4fs",
                    timestamp - self._last_frame_timestamp,
                )
            if self._last_frame_timestamp is not None and timestamp <= self._last_frame_timestamp + 1e-9:
                timestamp = self._last_frame_timestamp + 1e-6

            self._frame_cache.append(human_frame, timestamp, fps_timestamp=timestamp)
            self._last_raw_body_joints = body_joints.copy()
            self._last_frame_timestamp = timestamp
            self._last_source_seq = int(getattr(frame, "seq", self._last_source_seq or -1))

        self._frame_ready.set()
        return True

    def _accept_controller_snapshot(self, frame: Any, *, timestamp: float) -> None:
        seq = int(getattr(frame, "seq", self._last_source_seq or -1))
        controllers = getattr(frame, "controllers", None)
        snapshot = PicoControllerSnapshot(
            left=self._read_controller_state(None if controllers is None else getattr(controllers, "left", None)),
            right=self._read_controller_state(None if controllers is None else getattr(controllers, "right", None)),
            timestamp_s=float(timestamp),
            seq=seq,
        )
        with self._lock:
            self._controller_snapshot = snapshot

    def _poll_control_events(self, frame: Any, *, timestamp: float) -> bool:
        if self._pause_button_path is None:
            return False

        side, button_name = self._pause_button_path
        controllers = getattr(frame, "controllers", None)
        controller = None if controllers is None else getattr(controllers, side, None)
        buttons = {} if controller is None else getattr(controller, "buttons", {}) or {}
        pressed = bool(buttons.get(button_name, False))
        emitted = False
        if pressed and not self._last_pause_button_pressed:
            if (
                self._last_pause_toggle_timestamp is None
                or timestamp - self._last_pause_toggle_timestamp >= self._pause_debounce_s - 1e-9
            ):
                with self._lock:
                    self._pending_control_events.append(
                        ControlEvent(
                            event_type=ControlEventType.TOGGLE_PAUSE,
                            source=f"pico4:{self._pause_button}",
                            timestamp_s=float(timestamp),
                        )
                    )
                self._last_pause_toggle_timestamp = float(timestamp)
                emitted = True
        self._last_pause_button_pressed = pressed
        return emitted

    @staticmethod
    def _resolve_button_path(pause_button: str | None) -> tuple[str, str] | None:
        if pause_button is None:
            return None
        return _PAUSE_BUTTON_MAP.get(pause_button)

    @staticmethod
    def _read_controller_state(controller: Any) -> PicoControllerState:
        axis = {} if controller is None else getattr(controller, "axis", {}) or {}
        return PicoControllerState(
            raw=bool(False if controller is None else getattr(controller, "raw", False)),
            grip=float(axis.get("grip", 0.0)),
            trigger=float(axis.get("trigger", 0.0)),
            present=controller is not None,
        )

    @staticmethod
    def _convert_body_joints_to_frame(body_joints: NDArray[np.float64]) -> HumanFrame:
        body_joints = Pico4InputProvider._normalize_pico_bridge_body_joints(body_joints)
        body_pose_dict: dict[str, list] = {}
        for i, joint_name in enumerate(BODY_JOINT_NAMES):
            pos = [body_joints[i][0], body_joints[i][1], body_joints[i][2]]
            # pico_bridge returns [x, y, z, qx, qy, qz, qw].
            rot = [body_joints[i][6], body_joints[i][3], body_joints[i][4], body_joints[i][5]]
            body_pose_dict[joint_name] = [pos, rot]

        body_pose_dict = _coordinate_transform_input(body_pose_dict)

        result: HumanFrame = {}
        for name, (pos, quat) in body_pose_dict.items():
            result[name] = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        return result

    @staticmethod
    def _normalize_pico_bridge_body_joints(body_joints: NDArray[np.float64]) -> NDArray[np.float64]:
        """Match Teleopit's calibrated Pico body-pose convention."""
        converted = np.array(body_joints, dtype=np.float64, copy=True)
        converted[:, 2] *= -1.0
        converted[:, 5] *= -1.0
        converted[:, 6] *= -1.0
        return converted
