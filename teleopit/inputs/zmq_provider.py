"""ZMQ SUB input provider for onboard sim2real.

Subscribes to a ZMQ PUB socket that publishes raw Pico4 body tracking
frames (already coordinate-transformed into Teleopit space). Implements
the same interface as ``Pico4InputProvider`` / ``UDPBVHInputProvider``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Dict, Tuple

import msgpack
import numpy as np
from numpy.typing import NDArray

from teleopit.sim.reference_motion import interpolate_human_frames

logger = logging.getLogger(__name__)

HumanFrame = Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]


class ZMQInputProvider:
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
    ) -> None:
        import zmq

        self._human_format = human_format
        self._timeout = timeout
        self._topic = topic

        # ZMQ SUB socket
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s recv timeout for shutdown checks
        self._sock.setsockopt(zmq.RCVHWM, 5)  # limit receive queue to prevent stale-frame accumulation
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.subscribe(topic.encode("utf-8"))

        # Thread-safe state: deque buffer of (frame, timestamp) pairs
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._buffer: deque[tuple[HumanFrame, float]] = deque(maxlen=max(buffer_size, 2))
        self._fps_timestamps: deque[float] = deque(maxlen=30)  # source timestamps for fps measurement
        self._frame_seq: int = 0  # mirrors source _seq (only increments on real new frames)
        self._running = True

        # Clock offset: maps publisher monotonic → local monotonic (set once on first frame)
        self._clock_offset: float | None = None

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
            if len(self._fps_timestamps) < 2:
                return 30.0
            oldest = self._fps_timestamps[0]
            newest = self._fps_timestamps[-1]
            dt = newest - oldest
            if dt <= 0:
                return 30.0
            return (len(self._fps_timestamps) - 1) / dt

    @property
    def human_format(self) -> str:
        return self._human_format

    def is_available(self) -> bool:
        """True while the receiver thread is alive."""
        return self._running and self._thread.is_alive()

    def get_frame(self) -> HumanFrame:
        """Return the latest received frame.

        The first call blocks until the first ZMQ message arrives (up to
        ``timeout`` seconds).  Subsequent calls return immediately with the
        most recent frame.
        """
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            if not self._buffer:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
            return self._buffer[-1][0]

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        """Return the latest frame together with timestamp and sequence."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            if not self._buffer:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
            frame, ts = self._buffer[-1]
            return frame, ts, self._frame_seq

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Sample a delayed interpolated frame from the receive buffer."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            buf = list(self._buffer)

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
                raw = self._sock.recv()
                # Single frame: "<topic> <msgpack payload>"
                sep = raw.index(b" ")
                payload = msgpack.unpackb(raw[sep + 1:], raw=False)
                source_ts = payload.pop("_ts", None)
                source_seq = payload.pop("_seq", None)
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

            # Map publisher timestamp to local clock domain for correct
            # inter-frame spacing even when ZMQ queues messages.
            # Offset is set once on first frame; never auto-reset to avoid
            # masking staleness from queued old packets.
            if source_ts is not None:
                if self._clock_offset is None:
                    self._clock_offset = local_ts - source_ts
                buffer_ts = source_ts + self._clock_offset
            else:
                buffer_ts = local_ts  # fallback for publishers without _ts

            # Only count genuinely new frames (skip heartbeat duplicates)
            is_new_frame = source_seq is None or source_seq != self._frame_seq

            with self._lock:
                self._buffer.append((frame, buffer_ts))
                if is_new_frame:
                    self._fps_timestamps.append(source_ts if source_ts is not None else local_ts)
                    self._frame_seq = int(source_seq) if source_seq is not None else self._frame_seq + 1
            if not self._frame_ready.is_set():
                self._frame_ready.set()

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
