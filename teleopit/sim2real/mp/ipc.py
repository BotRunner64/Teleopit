"""Small-message IPC helpers for multiprocess sim2real."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
import time
from typing import Any

import zmq


BODY_TOPIC = "body"
HAND_TOPIC = "hand"
CONTROLLER_TOPIC = "controller"
CONTROL_EVENTS_TOPIC = "control_events"
REFERENCE_TOPIC = "reference"
MODE_TOPIC = "mode"
VIDEO_TOPIC = "video"
HEALTH_TOPIC = "health"
COMMAND_TOPIC = "command"


@dataclass(frozen=True)
class Sim2RealIpcEndpoints:
    body_pub: str
    hand_pub: str
    controller_pub: str
    control_events_pub: str
    reference_pub: str
    mode_pub: str
    video_pub: str
    health_pub: str
    command_pub: str


def default_endpoints(*, host: str = "127.0.0.1", base_port: int = 39700) -> Sim2RealIpcEndpoints:
    """Return deterministic localhost TCP endpoints for one sim2real runtime."""
    prefix = f"tcp://{host}:"
    return Sim2RealIpcEndpoints(
        body_pub=f"{prefix}{base_port}",
        hand_pub=f"{prefix}{base_port + 1}",
        controller_pub=f"{prefix}{base_port + 2}",
        control_events_pub=f"{prefix}{base_port + 3}",
        reference_pub=f"{prefix}{base_port + 4}",
        mode_pub=f"{prefix}{base_port + 5}",
        video_pub=f"{prefix}{base_port + 6}",
        health_pub=f"{prefix}{base_port + 7}",
        command_pub=f"{prefix}{base_port + 8}",
    )


class ZmqPublisher:
    """Topic publisher with low watermarks for realtime latest-only streams."""

    def __init__(self, endpoint: str, *, context: zmq.Context[Any] | None = None) -> None:
        self._own_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._endpoint = endpoint
        # Give subscribers a short chance to connect during process startup.
        time.sleep(0.05)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def publish(self, topic: str, payload: object) -> None:
        try:
            self._socket.send_multipart(
                [topic.encode("utf-8"), pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            # Realtime streams are latest-only; dropping beats backpressure.
            return

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._own_context:
            self._context.term()


class LatestSubscriber:
    """Subscriber that drains all pending messages and returns only the latest."""

    def __init__(
        self,
        endpoint: str,
        topic: str,
        *,
        context: zmq.Context[Any] | None = None,
    ) -> None:
        self._own_context = context is None
        self._context = zmq.Context() if context is None else context
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.connect(endpoint)
        self._topic = topic

    def recv_latest(self) -> object | None:
        latest: object | None = None
        while True:
            try:
                topic_raw, payload_raw = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return latest
            topic = topic_raw.decode("utf-8")
            if topic != self._topic:
                continue
            latest = pickle.loads(payload_raw)

    def close(self) -> None:
        self._socket.close(linger=0)
        if self._own_context:
            self._context.term()
