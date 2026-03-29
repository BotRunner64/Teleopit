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

logger = logging.getLogger(__name__)


def _serialize_frame(frame: dict) -> bytes:
    """Convert HumanFrame to msgpack bytes."""
    payload = {}
    for name, (pos, quat) in frame.items():
        payload[name] = [pos.tolist(), quat.tolist()]
    return msgpack.packb(payload, use_bin_type=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish Pico4 data over ZMQ")
    parser.add_argument("--bind", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ PUB port (default: 5555)")
    parser.add_argument("--topic", default="pico4", help="ZMQ topic (default: pico4)")
    parser.add_argument("--fps", type=float, default=30.0, help="Publish rate (default: 30)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    logger.info("ZMQ PUB bound on tcp://%s:%d topic=%s", args.bind, args.port, args.topic)

    provider = Pico4InputProvider(human_format="xrobot")
    topic_bytes = args.topic.encode("utf-8")
    dt = 1.0 / args.fps
    seq = 0

    print(f"Publishing Pico4 data on tcp://{args.bind}:{args.port} at {args.fps:.0f}fps...")
    print("Waiting for Pico4 body tracking data...")
    try:
        while True:
            t0 = time.monotonic()
            frame = provider.get_frame()
            payload = _serialize_frame(frame)
            sock.send_multipart([topic_bytes, payload])
            seq += 1
            if seq % 300 == 0:
                print(f"  Published {seq} frames")
            elapsed = time.monotonic() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print(f"\nStopped after {seq} frames.")
    finally:
        provider.close()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
