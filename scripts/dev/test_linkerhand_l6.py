#!/usr/bin/env python3
"""Exercise LinkerHand L6 open/close motion to verify hardware connectivity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_PATH = REPO_ROOT / "third_party" / "linkerhand-python-sdk"
if SDK_PATH.exists():
    sys.path.insert(0, str(SDK_PATH))


THUMB_YAW_DEFAULT = 10
OPEN_POSE = [250, THUMB_YAW_DEFAULT, 250, 250, 250, 250]
CLOSE_POSE = [79, THUMB_YAW_DEFAULT, 0, 0, 0, 0]
DEFAULT_SPEED = [50, 50, 50, 50, 50, 50]


def uint8(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 255:
        raise argparse.ArgumentTypeError("value must be in range 0-255")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def selected_hand_types(hand_type: str) -> tuple[str, ...]:
    if hand_type == "both":
        return ("left", "right")
    return (hand_type,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LinkerHand L6 open/close motion")
    parser.add_argument("--hand-type", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left-can", default="can0")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument(
        "--modbus",
        default="None",
        help='RS485 serial port such as /dev/ttyUSB0; "None" uses CAN',
    )
    parser.add_argument("--cycles", type=positive_int, default=3)
    parser.add_argument("--hold-s", type=positive_float, default=1.0)
    parser.add_argument("--thumb-yaw-center", type=uint8, default=THUMB_YAW_DEFAULT)
    parser.add_argument(
        "--speed",
        type=uint8,
        nargs=6,
        default=DEFAULT_SPEED,
        metavar=("THUMB_PITCH", "THUMB_YAW", "INDEX", "MIDDLE", "RING", "LITTLE"),
    )
    parser.add_argument(
        "--open-pose",
        type=uint8,
        nargs=6,
        default=OPEN_POSE,
        metavar=("THUMB_PITCH", "THUMB_YAW", "INDEX", "MIDDLE", "RING", "LITTLE"),
    )
    parser.add_argument(
        "--close-pose",
        type=uint8,
        nargs=6,
        default=CLOSE_POSE,
        metavar=("THUMB_PITCH", "THUMB_YAW", "INDEX", "MIDDLE", "RING", "LITTLE"),
    )
    args = parser.parse_args()
    args.open_pose[1] = args.thumb_yaw_center
    args.close_pose[1] = args.thumb_yaw_center
    return args


def send_all(hands: dict[str, object], pose: Sequence[int], *, label: str) -> None:
    print(f"{label}: {list(pose)}", flush=True)
    for hand_type, hand in hands.items():
        print(f"  {hand_type}", flush=True)
        hand.finger_move(pose=list(pose))


def main() -> None:
    args = parse_args()
    try:
        from LinkerHand.linker_hand_api import LinkerHandApi
    except ImportError as exc:
        raise SystemExit(
            "LinkerHand SDK import failed. Run: "
            "git submodule update --init third_party/linkerhand-python-sdk && "
            "pip install -e third_party/linkerhand-python-sdk"
        ) from exc

    hand_types = selected_hand_types(args.hand_type)
    can_channels = {"left": args.left_can, "right": args.right_can}
    hands: dict[str, object] = {}

    print(
        "Testing LinkerHand L6 | "
        f"hands={','.join(hand_types)} | "
        f"can={','.join(f'{hand}:{can_channels[hand]}' for hand in hand_types)} | "
        f"modbus={args.modbus}",
        flush=True,
    )
    try:
        for hand_type in hand_types:
            hand = LinkerHandApi(
                hand_joint="L6",
                hand_type=hand_type,
                modbus=args.modbus,
                can=can_channels[hand_type],
            )
            hand.set_speed(speed=list(args.speed))
            hands[hand_type] = hand

        send_all(hands, args.open_pose, label="startup open")
        time.sleep(args.hold_s)
        for cycle in range(args.cycles):
            print(f"cycle {cycle + 1}/{args.cycles}", flush=True)
            send_all(hands, args.close_pose, label="close")
            time.sleep(args.hold_s)
            send_all(hands, args.open_pose, label="open")
            time.sleep(args.hold_s)
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        if hands:
            send_all(hands, args.open_pose, label="exit open")
            time.sleep(0.2)
            print(
                "Exit cleanup intentionally leaves CAN interfaces up to avoid SDK network-down noise.",
                flush=True,
            )


if __name__ == "__main__":
    main()
