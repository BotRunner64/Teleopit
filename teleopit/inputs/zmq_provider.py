"""ZMQ SUB input provider for onboard sim2real.

Subscribes to a ZMQ PUB socket that publishes raw Pico4 body tracking
frames (already coordinate-transformed into Teleopit space). Implements
the same interface as ``Pico4InputProvider`` / ``UDPBVHInputProvider``.
"""

from __future__ import annotations

import logging
import threading
import time
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
    """

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 5555,
        topic: str = "pico4",
        human_format: str = "xrobot",
        timeout: float = 30.0,
    ) -> None:
        import zmq

        self._human_format = human_format
        self._timeout = timeout
        self._topic = topic

        # ZMQ SUB socket
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, 2000)  # 2s recv timeout for shutdown checks
        self._sock.setsockopt(zmq.CONFLATE, 1)  # keep only latest message
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.subscribe(topic.encode("utf-8"))

        # Thread-safe state (matches UDPBVHInputProvider / Pico4InputProvider pattern)
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._previous_frame: HumanFrame | None = None
        self._latest_frame: HumanFrame | None = None
        self._previous_timestamp: float | None = None
        self._latest_timestamp: float | None = None
        self._frame_seq: int = 0
        self._running = True

        # Receiver thread
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="zmq_input")
        self._thread.start()
        logger.info("ZMQInputProvider connecting to tcp://%s:%d topic=%s", host, port, topic)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def fps(self) -> int:
        """Pico4 body tracking runs at ~30 fps."""
        return 30

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
            if self._latest_frame is None:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
            return self._latest_frame

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        """Return the latest frame together with timestamp and sequence."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            if self._latest_frame is None or self._latest_timestamp is None:
                raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
            return self._latest_frame, float(self._latest_timestamp), int(self._frame_seq)

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Sample a delayed interpolated realtime frame from the receive buffer."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No ZMQ body data received within {self._timeout}s"
            )
        with self._lock:
            latest_frame = self._latest_frame
            latest_timestamp = self._latest_timestamp
            previous_frame = self._previous_frame
            previous_timestamp = self._previous_timestamp

        if latest_frame is None or latest_timestamp is None:
            raise RuntimeError("ZMQ frame buffer signaled ready without a latest frame")
        if (
            previous_frame is None
            or previous_timestamp is None
            or latest_timestamp <= previous_timestamp + 1e-6
        ):
            return latest_frame

        target_time = float(query_time_s - max(delay_s, 0.0))
        alpha = (target_time - previous_timestamp) / (latest_timestamp - previous_timestamp)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return interpolate_human_frames(previous_frame, latest_frame, alpha)

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
                parts = self._sock.recv_multipart()
                # parts[0] = topic bytes, parts[1] = msgpack payload
                payload = msgpack.unpackb(parts[1], raw=False)
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

            timestamp = time.monotonic()
            with self._lock:
                self._previous_frame = self._latest_frame
                self._previous_timestamp = self._latest_timestamp
                self._latest_frame = frame
                self._latest_timestamp = timestamp
                self._frame_seq += 1
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
