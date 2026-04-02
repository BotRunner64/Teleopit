"""Pico4 VR full-body motion capture input provider.

Uses xrobotoolkit_sdk to receive real-time body tracking data from a Pico4
headset. Outputs the same ``{joint_name: (position, quat)}`` dict format as
the BVH providers, so downstream retargeting is unchanged.
"""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, RealtimeInputPacket
from teleopit.inputs.rot_utils import quat_mul_np
from teleopit.sim.reference_motion import interpolate_human_frames

logger = logging.getLogger(__name__)

# Pre-compute the provider-space to Teleopit-space transform.
_INPUT_TO_TELEOPIT_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_INPUT_TO_TELEOPIT_QUAT = R.from_matrix(_INPUT_TO_TELEOPIT_MATRIX).as_quat(scalar_first=True)

# SMPL-X 24 body joint names from xrobotoolkit_sdk
BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

HumanFrame = Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]

_PICO_BUTTON_GETTERS: dict[str, str] = {
    "A": "get_A_button",
    "B": "get_B_button",
    "X": "get_X_button",
    "Y": "get_Y_button",
    "left_axis_click": "get_left_axis_click",
    "right_axis_click": "get_right_axis_click",
    "left_menu_button": "get_left_menu_button",
    "right_menu_button": "get_right_menu_button",
}


def _coordinate_transform_input(body_pose_dict: dict) -> dict:
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


class Pico4InputProvider:
    """Real-time input provider using Pico4 full-body tracking via xrobotoolkit_sdk.

    Implements the same interface as ``BVHInputProvider`` / ``UDPBVHInputProvider``
    so it plugs directly into the Teleopit pipeline.
    """

    def __init__(
        self,
        human_format: str = "xrobot",
        timeout: float = 60.0,
        buffer_size: int = 60,
        timestamp_gap_reset_s: float = 0.15,
        poll_sleep_s: float = 0.002,
        pause_button: str | None = "A",
        pause_debounce_s: float = 0.25,
    ) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise ImportError(
                "xrobotoolkit_sdk is required for Pico4 input. "
                "Install the Pico4 SDK or use a different input provider."
            ) from exc

        self._xrt = xrt
        self._xrt.init()
        self._human_format = human_format
        self._timeout = timeout
        self._closed = False
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._frame_cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size, fps_window=30)
        self._timestamp_gap_reset_s = float(timestamp_gap_reset_s)
        self._poll_sleep_s = max(float(poll_sleep_s), 0.0)
        self._pending_control_events: deque[ControlEvent] = deque()
        self._pause_button = None if pause_button in (None, "", "null") else str(pause_button)
        self._pause_debounce_s = max(float(pause_debounce_s), 0.0)
        self._pause_button_reader = self._resolve_button_reader(self._pause_button)
        self._last_pause_button_pressed = False
        self._last_pause_toggle_timestamp: float | None = None
        self._last_raw_body_poses: NDArray[np.float64] | None = None
        self._last_frame_timestamp: float | None = None
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="pico4_input")
        self._poll_thread.start()
        if self._pause_button is not None and self._pause_button_reader is None:
            logger.warning(
                "Pico4InputProvider pause button '%s' is unsupported by xrobotoolkit_sdk; pause events disabled",
                self._pause_button,
            )
        logger.info("Pico4InputProvider initialized (xrobotoolkit_sdk)")

    @property
    def fps(self) -> float:
        """Measured body tracking fps (falls back to 30.0 until enough samples)."""
        with self._lock:
            return self._frame_cache.fps()

    @property
    def human_format(self) -> str:
        return self._human_format

    def is_available(self) -> bool:
        """Whether the provider is healthy and can be polled.

        Returns ``True`` as long as the provider has not been closed.
        This matches ``UDPBVHInputProvider`` semantics where
        ``is_available()`` means "provider is alive" (not "data ready
        right now").  The actual blocking-until-data logic lives in
        ``get_frame()``.
        """
        return not self._closed and self._poll_thread.is_alive()

    def get_frame(self) -> HumanFrame:
        """Get the current body tracking frame.

        Blocks up to ``timeout`` seconds waiting for the first tracked
        frame, then returns the latest cached realtime frame.

        Returns:
            Dict mapping joint name to (position_3d, quaternion_wxyz) tuples
            after Teleopit's input-space transform.

        Raises:
            TimeoutError: If no body data arrives within the timeout.
        """
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest()

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        """Return the latest cached frame together with timestamp and sequence."""
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest_packet()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest_packet()

    def get_realtime_input_packet(self) -> RealtimeInputPacket[HumanFrame]:
        """Return the latest frame packet together with pending control events."""
        with self._lock:
            if len(self._frame_cache) > 0:
                frame, timestamp_s, seq = self._frame_cache.latest_packet()
                control_events = tuple(self._pending_control_events)
                self._pending_control_events.clear()
                return RealtimeInputPacket(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    seq=seq,
                    control_events=control_events,
                )
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            frame, timestamp_s, seq = self._frame_cache.latest_packet()
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return RealtimeInputPacket(
            frame=frame,
            timestamp_s=timestamp_s,
            seq=seq,
            control_events=control_events,
        )

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Sample a delayed interpolated realtime frame from the polling buffer."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
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
        """Mark provider as closed."""
        self._closed = True
        self._poll_thread.join(timeout=1.0)
        logger.info("Pico4InputProvider closed")

    def _poll_loop(self) -> None:
        while not self._closed:
            timestamp = time.monotonic()
            emitted_control_event = self._poll_control_events(timestamp=timestamp)
            if not self._xrt.is_body_data_available():
                time.sleep(self._poll_sleep_s if emitted_control_event else 0.001)
                continue

            try:
                body_poses = np.asarray(self._xrt.get_body_joints_pose(), dtype=np.float64)
            except Exception:
                logger.exception("Failed to read Pico4 body data")
                time.sleep(0.05)
                continue

            if not self._accept_body_poses(body_poses, timestamp=timestamp):
                time.sleep(self._poll_sleep_s if emitted_control_event else 0.001)
                continue

            time.sleep(self._poll_sleep_s)

    def _poll_control_events(self, *, timestamp: float) -> bool:
        if self._pause_button_reader is None:
            return False

        try:
            raw_value = self._pause_button_reader()
        except Exception:
            logger.exception("Failed to read Pico4 pause button '%s'; disabling pause events", self._pause_button)
            self._pause_button_reader = None
            return False

        pressed = self._normalize_button_value(raw_value)
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

    def _accept_body_poses(self, body_poses: NDArray[np.float64], *, timestamp: float) -> bool:
        if self._last_raw_body_poses is not None and np.array_equal(body_poses, self._last_raw_body_poses):
            return False

        frame = self._convert_body_poses_to_frame(body_poses)
        frame_timestamp = float(timestamp)

        with self._lock:
            if (
                self._last_frame_timestamp is not None
                and self._timestamp_gap_reset_s > 0.0
                and frame_timestamp - self._last_frame_timestamp > self._timestamp_gap_reset_s
            ):
                self._frame_cache.clear()
                logger.warning(
                    "Pico4InputProvider timestamp-gap reset | gap=%.4fs",
                    frame_timestamp - self._last_frame_timestamp,
                )
            if self._last_frame_timestamp is not None and frame_timestamp <= self._last_frame_timestamp + 1e-9:
                frame_timestamp = self._last_frame_timestamp + 1e-6

            self._frame_cache.append(frame, frame_timestamp, fps_timestamp=frame_timestamp)
            self._last_raw_body_poses = body_poses.copy()
            self._last_frame_timestamp = frame_timestamp

        self._frame_ready.set()
        return True

    def _resolve_button_reader(self, pause_button: str | None) -> Callable[[], object] | None:
        if pause_button is None:
            return None
        getter_name = _PICO_BUTTON_GETTERS.get(pause_button, pause_button)
        if not getter_name.startswith("get_"):
            getter_name = f"get_{getter_name}"
        getter = getattr(self._xrt, getter_name, None)
        if not callable(getter):
            return None
        return getter

    @staticmethod
    def _normalize_button_value(raw_value: object) -> bool:
        if isinstance(raw_value, np.ndarray):
            if raw_value.size <= 0:
                return False
            raw_value = raw_value.reshape(-1)[0]
        if isinstance(raw_value, (tuple, list)):
            if not raw_value:
                return False
            raw_value = raw_value[0]
        if isinstance(raw_value, (bool, np.bool_)):
            return bool(raw_value)
        if isinstance(raw_value, (int, float, np.integer, np.floating)):
            return float(raw_value) > 0.5
        return bool(raw_value)

    @staticmethod
    def _convert_body_poses_to_frame(body_poses: NDArray[np.float64]) -> HumanFrame:
        body_pose_dict: dict[str, list] = {}
        for i, joint_name in enumerate(BODY_JOINT_NAMES):
            pos = [body_poses[i][0], body_poses[i][1], body_poses[i][2]]
            # SDK returns [x,y,z,qx,qy,qz,qw] — convert to scalar-first [w,x,y,z]
            rot = [body_poses[i][6], body_poses[i][3], body_poses[i][4], body_poses[i][5]]
            body_pose_dict[joint_name] = [pos, rot]

        body_pose_dict = _coordinate_transform_input(body_pose_dict)

        result: HumanFrame = {}
        for name, (pos, quat) in body_pose_dict.items():
            result[name] = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        return result
