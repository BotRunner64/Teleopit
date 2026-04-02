"""Publish Pico4 body tracking data over ZMQ for onboard sim2real.

Runs on the upper machine (PC with Pico4 headset). The G1 onboard
computer subscribes to this data via ZMQInputProvider.

Usage:
    python scripts/zmq_pico4_publisher.py --port 5555
    python scripts/zmq_pico4_publisher.py --bind 0.0.0.0 --port 5555
"""

from __future__ import annotations

import argparse
import logging
import time

import msgpack
import zmq

from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.inputs.realtime_packet import ControlEvent

logger = logging.getLogger(__name__)


def _serialize_frame(
    frame: dict,
    source_ts: float,
    source_seq: int,
    control_events: tuple[ControlEvent, ...] = (),
) -> bytes:
    """Convert HumanFrame to msgpack bytes with source timestamp and sequence."""
    payload = {}
    for name, (pos, quat) in frame.items():
        payload[name] = [pos.tolist(), quat.tolist()]
    payload["_ts"] = source_ts
    payload["_seq"] = source_seq
    if control_events:
        payload["_control_events"] = [
            {
                "event_type": event.event_type.value,
                "source": event.source,
                "timestamp_s": event.timestamp_s,
            }
            for event in control_events
        ]
    return msgpack.packb(payload, use_bin_type=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Pico4 data over ZMQ")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port (default: 5555)")
    parser.add_argument("--topic", default="pico4", help="ZMQ topic (default: pico4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    logger.info("ZMQ PUB bound on tcp://%s:%d topic=%s", args.bind, args.port, args.topic)

    provider = Pico4InputProvider(human_format="xrobot")
    topic_bytes = args.topic.encode("utf-8")
    seq = 0
    last_frame_seq = -1
    last_send_time = 0.0
    heartbeat_interval = 0.1  # resend latest frame every 100ms when stationary

    print(f"Publishing Pico4 data on tcp://{args.bind}:{args.port} ...")
    print("Waiting for Pico4 body tracking data...")
    try:
        while True:
            packet = provider.get_realtime_input_packet()
            frame = packet.frame
            ts = packet.timestamp_s
            frame_seq = packet.seq
            control_events = packet.control_events
            now = time.monotonic()
            if frame_seq == last_frame_seq and not control_events and now - last_send_time < heartbeat_interval:
                time.sleep(0.001)
                continue
            last_frame_seq = frame_seq
            last_send_time = now
            payload = _serialize_frame(frame, ts, frame_seq, control_events)
            sock.send(topic_bytes + b" " + payload)
            seq += 1
            if seq % 300 == 0:
                fps = provider.fps
                print(f"  Published {seq} frames ({fps:.1f} fps)")
    except KeyboardInterrupt:
        print(f"\nStopped after {seq} frames.")
    finally:
        provider.close()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
