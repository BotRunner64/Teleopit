"""Realtime UDP BVH streaming input provider.

Receives BVH motion data over UDP (one packet = one frame of
whitespace-separated floats) and converts each frame to a HumanFrame
dict using the same processing logic as the offline BVH provider.

The hc_mocap skeleton definition is hardcoded — no reference BVH needed.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot

from teleopit.inputs.bvh_provider import process_single_bvh_frame
from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import (
    ControlEvent,
    HumanFrame,
    RealtimeInputPacket,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded hc_mocap skeleton definition (52 joints, 3-channel, zxy euler)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SkeletonDef:
    bone_names: list[str]
    bone_parents: NDArray[np.int32]
    offsets: NDArray[np.float64]
    euler_order: str
    channels: int
    scale_divisor: float


_HC_MOCAP_BONE_NAMES: list[str] = [
    "hc_Abdomen", "hc_Hip_L", "hc_Knee_L", "hc_Foot_L", "LeftToeBase",
    "hc_Hip_R", "hc_Knee_R", "hc_Foot_R", "RightToeBase", "Spine",
    "hc_Chest", "hc_Chest1", "LeftShoulder", "hc_Shoulder_L", "hc_Elbow_L",
    "hc_Hand_L", "hc_Index1_L", "hc_Index2_L", "hc_Index3_L", "hc_Middle1_L",
    "hc_Middle2_L", "hc_Middle3_L", "hc_Pinky1_L", "hc_Pinky2_L", "hc_Pinky3_L",
    "hc_Ring1_L", "hc_Ring2_L", "hc_Ring3_L", "hc_Thumb1_L", "hc_Thumb2_L",
    "hc_Thumb3_L", "neck", "hc_Head", "RightShoulder1", "hc_Shoulder_R",
    "hc_Elbow_R", "hc_Hand_R", "hc_Index1_R", "hc_Index2_R", "hc_Index3_R",
    "hc_Middle1_R", "hc_Middle2_R", "hc_Middle3_R", "hc_Pinky1_R", "hc_Pinky2_R",
    "hc_Pinky3_R", "hc_Ring1_R", "hc_Ring2_R", "hc_Ring3_R", "hc_Thumb1_R",
    "hc_Thumb2_R", "hc_Thumb3_R",
]

_HC_MOCAP_BONE_PARENTS: list[int] = [
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    15, 19, 20, 15, 22, 23, 15, 25, 26, 15, 28, 29, 11, 31, 11, 33, 34,
    35, 36, 37, 38, 36, 40, 41, 36, 43, 44, 36, 46, 47, 36, 49, 50,
]

# fmt: off
_HC_MOCAP_OFFSETS: list[list[float]] = [
    [-0.03, 0.0, 0.0], [-0.090058, -0.014508, -0.005261],
    [0.0, -0.42572, 0.0], [0.0, -0.401965, 0.0],
    [0.008324, -0.164266, -0.10654], [0.090058, -0.014508, -0.005261],
    [0.0, -0.42572, 0.0], [0.0, -0.401965, 0.0],
    [-0.008324, -0.164266, -0.10654], [0.0, 0.108057, -0.00891],
    [0.0, 0.005886, 0.192453], [0.0, 0.132667, 0.019818],
    [-0.03782, 0.121714, -0.007377], [0.0, 0.0, 0.161531],
    [0.0, -0.298399, 0.0], [0.0, -0.269751, 0.0],
    [-0.027655, -0.120344, -0.008508], [0.0, 0.0, 0.042875],
    [0.0, 0.0, 0.033938], [-0.001078, -0.123213, -0.003106],
    [0.0, 0.0, 0.046404], [0.0, 0.0, 0.036488],
    [0.040938, -0.105318, -0.013548], [0.0, 0.0, 0.03571],
    [0.000881, 0.0, 0.029796], [0.022144, -0.117419, -0.007786],
    [0.0, 0.0, 0.044193], [0.000638, 0.0, 0.034791],
    [-0.027649, -0.047882, -0.020461], [0.0, -0.038697, 0.0],
    [0.0, -0.040622, 0.0], [0.0, 0.163912, 0.023766],
    [0.0, -0.018972, 0.09095], [0.03782, 0.121711, -0.007377],
    [0.0, 0.0, 0.16153], [0.0, -0.298397, 0.0],
    [0.0, -0.269752, 0.0], [0.023345, -0.121506, -0.003388],
    [0.0, 0.0, 0.042879], [0.0, 0.0, 0.033935],
    [-0.002997, -0.123175, 0.003474], [0.0, 0.0, 0.046401],
    [0.0, 0.0, 0.036488], [-0.044907, -0.104415, -0.005811],
    [0.0, 0.0, 0.035707], [0.0, 0.0, 0.029808],
    [-0.026234, -0.116837, -0.000348], [0.0, 0.0, 0.044191],
    [0.0, 0.0, 0.034797], [0.025022, -0.04981, -0.019209],
    [0.0, -0.038698, 0.0], [0.0, -0.04062, 0.0],
]
# fmt: on

HC_MOCAP_SKELETON = _SkeletonDef(
    bone_names=_HC_MOCAP_BONE_NAMES,
    bone_parents=np.array(_HC_MOCAP_BONE_PARENTS, dtype=np.int32),
    offsets=np.array(_HC_MOCAP_OFFSETS, dtype=np.float64),
    euler_order="zxy",
    channels=3,
    scale_divisor=1.0,
)

_SKELETONS: dict[str, _SkeletonDef] = {
    "hc_mocap": HC_MOCAP_SKELETON,
}


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def _lerp_frames(frame_a: HumanFrame, frame_b: HumanFrame, alpha: float) -> HumanFrame:
    """Linearly interpolate two HumanFrame dicts (position lerp, quat slerp)."""
    from teleopit.retargeting.gmr.utils.lafan_vendor import utils

    result: HumanFrame = {}
    for bone in frame_b:
        if bone not in frame_a:
            result[bone] = frame_b[bone]
            continue
        pos_a, quat_a = frame_a[bone]
        pos_b, quat_b = frame_b[bone]
        pos = pos_a * (1.0 - alpha) + pos_b * alpha
        quat = utils.quat_slerp(quat_a, quat_b, alpha)
        result[bone] = (pos, quat)
    return result


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class UDPBVHInputProvider:
    """Realtime input provider that receives BVH frames over UDP.

    Implements the ``RealtimeInputProvider`` protocol.
    """

    def __init__(
        self,
        bvh_format: str = "hc_mocap",
        human_height: float = 1.75,
        udp_host: str = "",
        udp_port: int = 1118,
        udp_timeout: float = 30.0,
        buffer_size: int = 60,
    ) -> None:
        skel = _SKELETONS.get(bvh_format)
        if skel is None:
            raise ValueError(
                f"Unsupported bvh_format '{bvh_format}' for UDP streaming. "
                f"Supported: {list(_SKELETONS)}."
            )

        self._bone_names = skel.bone_names
        self._bone_parents = skel.bone_parents
        self._offsets = skel.offsets.copy()
        self._euler_order = skel.euler_order
        self._channels = skel.channels
        self._scale_divisor = skel.scale_divisor

        N = len(self._bone_names)
        if self._channels == 3:
            self._expected_floats = 3 + N * 3
        elif self._channels == 6:
            self._expected_floats = N * 6
        else:
            raise ValueError(f"Unsupported channel count: {self._channels}")

        # Coordinate transform: Y-up → Z-up
        self._rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        self._rotation_quat = Rot.from_matrix(self._rotation_matrix).as_quat(scalar_first=True)

        self._bvh_format = bvh_format
        self._human_format = bvh_format
        self._human_height = human_height
        self._udp_host = udp_host
        self._udp_port = udp_port
        self._udp_timeout = udp_timeout

        self._cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size)
        self._lock = threading.Lock()
        self._first_frame_event = threading.Event()
        self._running = True
        self._control_events: deque[ControlEvent] = deque()

        # Bind socket on main thread so port/address errors surface immediately.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(2.0)
        self._sock.bind((self._udp_host, self._udp_port))

        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()
        log.info("UDPBVHInputProvider listening on %s:%d", udp_host or "0.0.0.0", udp_port)

    # -- properties --

    @property
    def fps(self) -> float:
        return self._cache.fps()

    @property
    def bone_names(self) -> list[str]:
        return self._bone_names

    @property
    def bone_parents(self) -> NDArray[np.int32]:
        return self._bone_parents

    @property
    def human_format(self) -> str:
        return self._human_format

    @property
    def human_height(self) -> float:
        return self._human_height

    # -- InputProvider --

    def is_available(self) -> bool:
        return self._running and self._thread.is_alive()

    def get_frame(self) -> HumanFrame:
        self._first_frame_event.wait(timeout=self._udp_timeout)
        if not self._first_frame_event.is_set():
            raise TimeoutError(
                f"No UDP BVH data received within {self._udp_timeout}s on port {self._udp_port}"
            )
        with self._lock:
            return self._cache.latest()

    # -- RealtimeInputProvider --

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        self._first_frame_event.wait(timeout=self._udp_timeout)
        if not self._first_frame_event.is_set():
            raise TimeoutError(
                f"No UDP BVH data received within {self._udp_timeout}s on port {self._udp_port}"
            )
        with self._lock:
            return self._cache.latest_packet()

    def get_realtime_input_packet(self) -> RealtimeInputPacket[HumanFrame]:
        frame, ts, seq = self.get_frame_packet()
        with self._lock:
            events = tuple(self._control_events)
            self._control_events.clear()
        return RealtimeInputPacket(frame=frame, timestamp_s=ts, seq=seq, control_events=events)

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Return an interpolated frame for the requested time."""
        with self._lock:
            snap = self._cache.snapshot()

        if not snap:
            return self.get_frame()

        target = query_time_s - delay_s

        if len(snap) < 2:
            return snap[0][0]

        for i in range(len(snap) - 1, 0, -1):
            ts_b = snap[i][1]
            ts_a = snap[i - 1][1]
            if ts_a <= target <= ts_b:
                dt = ts_b - ts_a
                alpha = (target - ts_a) / dt if dt > 0 else 1.0
                return _lerp_frames(snap[i - 1][0], snap[i][0], alpha)

        if target <= snap[0][1]:
            return snap[0][0]
        return snap[-1][0]

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)
        log.info("UDPBVHInputProvider closed")

    # -- receiver thread --

    def _receiver_loop(self) -> None:
        try:
            while self._running:
                try:
                    data, _addr = self._sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    if self._running:
                        log.warning("UDP socket error, stopping receiver")
                    break

                self._process_packet(data)
        finally:
            self._sock.close()

    def _process_packet(self, data: bytes) -> None:
        try:
            text = data.decode("utf-8").strip()
            if not text:
                return
            floats = np.fromstring(text, sep=" ", dtype=np.float64)
        except (UnicodeDecodeError, ValueError) as exc:
            log.warning("Malformed UDP packet: %s", exc)
            return

        if len(floats) != self._expected_floats:
            log.warning(
                "Expected %d floats, got %d — skipping frame",
                self._expected_floats,
                len(floats),
            )
            return

        frame = process_single_bvh_frame(
            data_floats=floats,
            offsets=self._offsets,
            bone_names=self._bone_names,
            bone_parents=self._bone_parents,
            euler_order=self._euler_order,
            rotation_quat=self._rotation_quat,
            rotation_matrix=self._rotation_matrix,
            format=self._bvh_format,
            scale_divisor=self._scale_divisor,
            channels=self._channels,
        )

        now = time.monotonic()
        with self._lock:
            self._cache.append(frame, now)

        if not self._first_frame_event.is_set():
            self._first_frame_event.set()
            log.info("First UDP BVH frame received")
