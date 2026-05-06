"""ZMQ SUB input provider for onboard sim2real.

Subscribes to a ZMQ PUB socket that publishes raw Pico4 body tracking
frames (already coordinate-transformed into Teleopit space). Implements
the same interface as ``Pico4InputProvider`` for the shared realtime
motion-control pipeline.
"""

from __future__ import annotations

from collections import deque
import logging
import threading
import time
from typing import Dict, Tuple

import msgpack
import numpy as np
from numpy.typing import NDArray

from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, BODY_JOINT_PARENTS
from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, HumanFrame, RealtimeInputPacket
from teleopit.sim.reference_motion import interpolate_human_frames
from teleopit.interfaces import RealtimeInputProvider

logger = logging.getLogger(__name__)


class ZMQInputProvider(RealtimeInputProvider):
    """Receives Pico4 body tracking frames over ZMQ SUB.

    Parameters
    ----------
    host : str
        ZMQ publisher IP address (e.g. ``"192.168.1.100"``).
    port : int
        ZMQ publisher port (default 5555).
    topic : str
        ZMQ topic filter (default ``"pico4"``).
    human_format : str
        Format tag for retargeting (default ``"xrobot"``).
    timeout : float
        Seconds to wait for the first frame before raising.
    buffer_size : int
        Maximum number of frames to keep in the receive buffer.
    """

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 5555,
        topic: str = "pico4",
        human_format: str = "xrobot",
        timeout: float = 30.0,
        buffer_size: int = 60,
        conflate: bool = True,
        recv_hwm: int = 1,
        seq_gap_reset_threshold: int = 4,
    ) -> None:
        import zmq

        self._human_format = human_format
        self._timeout = timeout
        self._topic = topic
        self._conflate = bool(conflate)
        self._recv_hwm = max(int(recv_hwm), 1)
        self._seq_gap_reset_threshold = max(int(seq_gap_reset_threshold), 1)

        # ZMQ SUB socket
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s recv timeout for shutdown checks
        self._sock.setsockopt(zmq.RCVHWM, 1 if self._conflate else self._recv_hwm)
        if self._conflate:
            try:
                self._sock.setsockopt(zmq.CONFLATE, 1)
            except (AttributeError, zmq.ZMQError):
                logger.warning("ZMQ CONFLATE unavailable; falling back to manual queue draining")
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.subscribe(topic.encode("utf-8"))

        # Thread-safe state
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._frame_cache = RealtimeFrameCache[HumanFrame](buffer_size=buffer_size, fps_window=30)
        self._pending_control_events: deque[ControlEvent] = deque()
        self._running = True

        # Clock offset: maps publisher monotonic → local monotonic (set once on first frame)
        self._clock_offset: float | None = None
        self._last_source_seq: int | None = None
        self._last_buffer_ts: float | None = None

        # Receiver thread
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="zmq_input")
        self._thread.start()
        logger.info("ZMQInputProvider connecting to tcp://%s:%d topic=%s", host, port, topic)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Measured source fps (falls back to 30.0 until enough samples)."""
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
        """True while the receiver thread is alive."""
        return self._running and self._thread.is_alive()

    def get_frame(self) -> HumanFrame:
        """Return the latest received frame.

        The first call blocks until the first ZMQ message arrives (up to
        ``timeout`` seconds).  Subsequent calls return immediately with the
        most recent frame.
        """
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
            return self._frame_cache.latest()

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        """Return the latest frame together with timestamp and sequence."""
        with self._lock:
            if len(self._frame_cache) > 0:
                return self._frame_cache.latest_packet()
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
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
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            if len(self._frame_cache) <= 0:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
            frame, timestamp_s, seq = self._frame_cache.latest_packet()
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return RealtimeInputPacket(
            frame=frame,
            timestamp_s=timestamp_s,
            seq=seq,
            control_events=control_events,
        )

    def pop_control_events(self) -> tuple[ControlEvent, ...]:
        """Return pending control events without blocking on frame availability."""
        with self._lock:
            control_events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
        return control_events

    def has_frame(self) -> bool:
        """Whether at least one realtime frame is cached locally."""
        with self._lock:
            return len(self._frame_cache) > 0

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Sample a delayed interpolated frame from the receive buffer."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            buf = self._frame_cache.snapshot()

        if not buf:
            raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
        if len(buf) == 1:
            return buf[0][0]

        target_time = float(query_time_s - max(delay_s, 0.0))

        # Clamp to buffer bounds
        if target_time <= buf[0][1]:
            return buf[0][0]
        if target_time >= buf[-1][1]:
            return buf[-1][0]

        # Linear scan to find bracketing frames
        for i in range(1, len(buf)):
            older_frame, older_ts = buf[i - 1]
            newer_frame, newer_ts = buf[i]
            if target_time <= newer_ts:
                dt = newer_ts - older_ts
                if dt <= 1e-6:
                    return newer_frame
                alpha = float(np.clip((target_time - older_ts) / dt, 0.0, 1.0))
                return interpolate_human_frames(older_frame, newer_frame, alpha)

        # Fallback: return latest
        return buf[-1][0]

    def close(self) -> None:
        """Stop the receiver thread and close the socket."""
        self._running = False
        self._thread.join(timeout=3)
        try:
            self._sock.close()
            self._ctx.term()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Background receiver
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        import zmq

        while self._running:
            try:
                raw, drained_count = self._recv_latest_raw()
                # Single frame: "<topic> <msgpack payload>"
                sep = raw.index(b" ")
                payload = msgpack.unpackb(raw[sep + 1:], raw=False)
                source_ts = self._normalize_source_ts(payload.pop("_ts", None))
                source_seq = self._normalize_source_seq(payload.pop("_seq", None))
                control_events = self._normalize_control_events(
                    payload.pop("_control_events", payload.pop("control_events", None))
                )
                frame = self._deserialize_frame(payload)
            except zmq.Again:  # recv timeout — check shutdown flag
                continue
            except zmq.ZMQError:
                if self._running:
                    logger.warning("ZMQ socket error, stopping receiver")
                break
            except Exception:
                logger.exception("Failed to process ZMQ frame")
                continue

            local_ts = time.monotonic()
            self._process_packet(
                frame,
                source_ts=source_ts,
                source_seq=source_seq,
                local_ts=local_ts,
                drained_count=drained_count,
                control_events=control_events,
            )
            if not self._frame_ready.is_set():
                self._frame_ready.set()

    def _recv_latest_raw(self) -> tuple[bytes, int]:
        import zmq

        raw = self._sock.recv()
        drained_count = 0
        while True:
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
                drained_count += 1
            except zmq.Again:
                break
        return raw, drained_count

    def _process_packet(
        self,
        frame: HumanFrame,
        *,
        source_ts: float | None,
        source_seq: int | None,
        local_ts: float,
        drained_count: int = 0,
        control_events: tuple[ControlEvent, ...] = (),
    ) -> None:
        buffer_ts = self._map_buffer_timestamp(source_ts, local_ts)

        if drained_count > 0:
            logger.debug(
                "ZMQInputProvider drained %d queued packets before processing seq=%s",
                drained_count,
                "none" if source_seq is None else source_seq,
            )

        with self._lock:
            if source_seq is not None and self._last_source_seq is not None:
                if source_seq <= self._last_source_seq:
                    return
                seq_gap = source_seq - self._last_source_seq
                if seq_gap > self._seq_gap_reset_threshold:
                    self._frame_cache.clear()
                    logger.warning(
                        "ZMQInputProvider seq-gap catch-up | last_seq=%d | new_seq=%d | gap=%d",
                        self._last_source_seq,
                        source_seq,
                        seq_gap,
                    )
            elif self._last_buffer_ts is not None and buffer_ts <= self._last_buffer_ts + 1e-9:
                return

            if self._last_buffer_ts is not None and buffer_ts <= self._last_buffer_ts + 1e-9:
                buffer_ts = self._last_buffer_ts + 1e-6

            self._frame_cache.append(
                frame,
                buffer_ts,
                fps_timestamp=source_ts if source_ts is not None else local_ts,
                source_seq=source_seq,
            )
            self._pending_control_events.extend(control_events)
            self._last_buffer_ts = buffer_ts
            if source_seq is not None:
                self._last_source_seq = source_seq

    def _map_buffer_timestamp(self, source_ts: float | None, local_ts: float) -> float:
        if source_ts is None:
            return float(local_ts)
        if self._clock_offset is None:
            self._clock_offset = float(local_ts) - float(source_ts)
        return float(source_ts) + float(self._clock_offset)

    @staticmethod
    def _normalize_source_ts(raw_value: object) -> float | None:
        if raw_value in (None, "", "null"):
            return None
        value = float(raw_value)
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _normalize_source_seq(raw_value: object) -> int | None:
        if raw_value in (None, "", "null"):
            return None
        if isinstance(raw_value, bool):
            return int(raw_value)
        return int(raw_value)

    @staticmethod
    def _normalize_control_events(raw_value: object) -> tuple[ControlEvent, ...]:
        if raw_value in (None, "", "null"):
            return ()

        values = raw_value if isinstance(raw_value, list) else [raw_value]
        events: list[ControlEvent] = []
        for item in values:
            if isinstance(item, str):
                if item == ControlEventType.TOGGLE_PAUSE.value:
                    events.append(
                        ControlEvent(
                            event_type=ControlEventType.TOGGLE_PAUSE,
                            source="zmq",
                            timestamp_s=None,
                        )
                    )
                continue

            if not isinstance(item, dict):
                continue

            raw_type = item.get("event_type", item.get("type"))
            if raw_type != ControlEventType.TOGGLE_PAUSE.value:
                continue
            raw_timestamp = item.get("timestamp_s", item.get("timestamp"))
            timestamp_s = None if raw_timestamp in (None, "", "null") else float(raw_timestamp)
            events.append(
                ControlEvent(
                    event_type=ControlEventType.TOGGLE_PAUSE,
                    source=str(item.get("source", "zmq")),
                    timestamp_s=timestamp_s,
                )
            )
        return tuple(events)

    @staticmethod
    def _deserialize_frame(payload: dict) -> HumanFrame:
        """Convert msgpack dict back to HumanFrame."""
        frame: HumanFrame = {}
        for name, (pos, quat) in payload.items():
            frame[name] = (
                np.asarray(pos, dtype=np.float64),
                np.asarray(quat, dtype=np.float64),
            )
        return frame
