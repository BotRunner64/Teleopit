#!/usr/bin/env python3
"""Exercise LinkerHand dexterous-hand control modes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SDK_PATH = REPO_ROOT / "third_party" / "linkerhand-python-sdk"
if SDK_PATH.exists():
    sys.path.insert(0, str(SDK_PATH))
SOMEHAND_SRC_PATH = REPO_ROOT / "third_party" / "somehand" / "src"
if SOMEHAND_SRC_PATH.exists():
    sys.path.insert(0, str(SOMEHAND_SRC_PATH))

from teleopit.inputs.pico4_provider import (  # noqa: E402
    Pico4InputProvider,
)
from teleopit.sim2real.hands.linkerhand_l6 import VR_HAND_POSE_SPEED, build_linkerhand_l6  # noqa: E402
from teleopit.sim2real.hands.linkerhand_o6 import build_linkerhand_o6  # noqa: E402


THUMB_YAW_DEFAULT = 10
OPEN_POSE = [250, THUMB_YAW_DEFAULT, 250, 250, 250, 250]
CLOSE_POSE = [79, THUMB_YAW_DEFAULT, 0, 0, 0, 0]
DEFAULT_SPEED = [50, 50, 50, 50, 50, 50]
O6_OPEN_POSE = [250, 250, 250, 250, 250, 250]
O6_CLOSE_POSE = [86, 73, 118, 111, 110, 111]
O6_DEFAULT_SPEED = [255, 255, 255, 255, 255, 255]
DEFAULT_SOMEHAND_CONFIG_PATH = "third_party/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml"
OPEN_CLOSE_HOLD_S = 1.0
GRIPPER_RATE_HZ = 30.0
VR_HAND_POSE_RATE_HZ = 60.0
PICO_START_TIMEOUT_S = 60.0
FRAME_TIMEOUT_S = 0.3
TRIGGER_DEADZONE = 0.05
DEADMAN_THRESHOLD = 0.5


def selected_hand_types(hand_type: str) -> tuple[str, ...]:
    if hand_type == "both":
        return ("left", "right")
    return (hand_type,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LinkerHand dexterous-hand control modes")
    parser.add_argument(
        "--driver",
        choices=["linkerhand_l6", "linkerhand_o6"],
        default="linkerhand_l6",
        help="Hand driver to test. O6 currently supports open_close and gripper only.",
    )
    parser.add_argument(
        "--mode",
        choices=["open_close", "gripper", "vr_hand_pose"],
        default="open_close",
        help=(
            "open_close sends fixed poses directly; gripper reads real Pico controller "
            "grip/trigger input; vr_hand_pose reads real Pico hand pose input through Teleopit "
            "and uses somehand only for hand retargeting."
        ),
    )
    parser.add_argument("--hand-type", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left-can", default="can0")
    parser.add_argument("--right-can", default="can1")
    parser.add_argument(
        "--modbus",
        default="None",
        help='RS485 serial port such as /dev/ttyUSB0; "None" uses CAN',
    )
    args = parser.parse_args()
    if args.driver == "linkerhand_o6" and args.mode == "vr_hand_pose":
        raise SystemExit("hands.driver=linkerhand_o6 supports only --mode open_close or gripper")
    args.speed = list(O6_DEFAULT_SPEED if args.driver == "linkerhand_o6" else DEFAULT_SPEED)
    args.open_pose = list(O6_OPEN_POSE if args.driver == "linkerhand_o6" else OPEN_POSE)
    args.close_pose = list(O6_CLOSE_POSE if args.driver == "linkerhand_o6" else CLOSE_POSE)
    return args


def make_config(args: argparse.Namespace, *, mode: str) -> dict[str, object]:
    speed = VR_HAND_POSE_SPEED if mode == "vr_hand_pose" else args.speed
    rate_hz = VR_HAND_POSE_RATE_HZ if mode == "vr_hand_pose" else GRIPPER_RATE_HZ
    driver_section = "linkerhand_o6" if args.driver == "linkerhand_o6" else "linkerhand_l6"
    driver_cfg = {
        "left_can": args.left_can,
        "right_can": args.right_can,
        "modbus": args.modbus,
        "trigger_deadzone": TRIGGER_DEADZONE,
        "deadman_threshold": DEADMAN_THRESHOLD,
        "speed": list(speed),
        "open_pose": list(args.open_pose),
        "close_pose": list(args.close_pose),
        "print_input": False,
    }
    if args.driver == "linkerhand_l6":
        driver_cfg["thumb_yaw_center"] = THUMB_YAW_DEFAULT
    return {
        "input": {"provider": "pico4"},
        "hands": {
            "enabled": True,
            "driver": args.driver,
            "mode": mode,
            "sides": list(selected_hand_types(args.hand_type)),
            "rate_hz": rate_hz,
            "frame_timeout_s": FRAME_TIMEOUT_S,
            driver_section: driver_cfg,
            "somehand": {
                "config_path": DEFAULT_SOMEHAND_CONFIG_PATH,
                "rate_hz": VR_HAND_POSE_RATE_HZ,
                "max_iterations": 12,
                "temporal_filter_alpha": 1.0,
                "output_alpha": 1.0,
            },
        },
    }


def build_driver_runtime(config: dict[str, object], *, driver: str):
    if driver == "linkerhand_o6":
        return build_linkerhand_o6(config)
    return build_linkerhand_l6(config)


def send_all(hands: dict[str, object], pose: Sequence[int], *, label: str) -> None:
    print(f"{label}: {list(pose)}", flush=True)
    for hand_type, hand in hands.items():
        print(f"  {hand_type}", flush=True)
        hand.finger_move(pose=list(pose))


def make_pico_provider() -> Pico4InputProvider:
    return Pico4InputProvider(
        timeout=PICO_START_TIMEOUT_S,
        pause_button=None,
        bridge_host="0.0.0.0",
        bridge_port=63901,
        bridge_discovery=True,
        bridge_advertise_ip=None,
        bridge_start_timeout=10.0,
        bridge_video=None,
        bridge_video_enabled=False,
    )


def run_live_until_done(
    runtime: object,
    *,
    provider: Pico4InputProvider,
    mode_label: str,
    rate_hz: float,
) -> None:
    last_seq: int | None = None
    print(f"Running {mode_label}; press Ctrl-C to stop.", flush=True)
    while True:
        now_s = time.monotonic()
        controller_snapshot = provider.get_controller_snapshot()
        hand_snapshot = provider.get_hand_snapshot()
        runtime.tick(
            controller_snapshot=controller_snapshot,
            hand_snapshot=hand_snapshot,
            active=True,
            now_s=now_s,
        )
        snapshot = controller_snapshot if mode_label == "gripper" else hand_snapshot
        if snapshot is not None and snapshot.seq != last_seq:
            last_seq = snapshot.seq
            age_ms = max((now_s - snapshot.timestamp_s) * 1000.0, 0.0)
            print(f"  pico seq={snapshot.seq} age={age_ms:.1f}ms", flush=True)
        time.sleep(max(1.0 / rate_hz, 0.001))


def run_open_close(args: argparse.Namespace) -> None:
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
        f"Testing {args.driver} | "
        f"hands={','.join(hand_types)} | "
        f"can={','.join(f'{hand}:{can_channels[hand]}' for hand in hand_types)} | "
        f"modbus={args.modbus}",
        flush=True,
    )
    try:
        for hand_type in hand_types:
            hand = LinkerHandApi(
                hand_joint="O6" if args.driver == "linkerhand_o6" else "L6",
                hand_type=hand_type,
                modbus=args.modbus,
                can=can_channels[hand_type],
            )
            hand.set_speed(speed=list(args.speed))
            hands[hand_type] = hand

        send_all(hands, args.open_pose, label="startup open")
        time.sleep(OPEN_CLOSE_HOLD_S)
        cycle = 0
        while True:
            cycle += 1
            print(f"cycle {cycle}", flush=True)
            send_all(hands, args.close_pose, label="close")
            time.sleep(OPEN_CLOSE_HOLD_S)
            send_all(hands, args.open_pose, label="open")
            time.sleep(OPEN_CLOSE_HOLD_S)
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


def run_gripper(args: argparse.Namespace) -> None:
    config = make_config(args, mode="gripper")
    provider = make_pico_provider()
    device, mapper = build_driver_runtime(config, driver=args.driver)
    from teleopit.sim2real.hands.worker import HandRuntime
    runtime = HandRuntime(device, mapper)

    print(
        "Testing hands.mode=gripper with real Pico controller input. "
        "Hold grip above the deadman threshold, then use trigger to close/open.",
        flush=True,
    )
    try:
        runtime.start()
        run_live_until_done(
            runtime,
            provider=provider,
            mode_label="gripper",
            rate_hz=GRIPPER_RATE_HZ,
        )
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        runtime.tick(controller_snapshot=None, hand_snapshot=None, active=False)
        runtime.close()
        provider.close()


def run_vr_hand_pose(args: argparse.Namespace) -> None:
    if args.hand_type != "both":
        raise SystemExit("hands.mode=vr_hand_pose currently requires --hand-type both")

    config = make_config(args, mode="vr_hand_pose")
    provider = make_pico_provider()
    device, mapper = build_driver_runtime(config, driver=args.driver)
    from teleopit.sim2real.hands.worker import HandRuntime
    runtime = HandRuntime(device, mapper)

    print(
        "Testing hands.mode=vr_hand_pose with real Pico hand-pose input. "
        "Enable Pico hand tracking and move both hands; start with the robot clear of contacts.",
        flush=True,
    )
    try:
        runtime.start()
        run_live_until_done(
            runtime,
            provider=provider,
            mode_label="vr_hand_pose",
            rate_hz=VR_HAND_POSE_RATE_HZ,
        )
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        runtime.tick(controller_snapshot=None, hand_snapshot=None, active=False)
        runtime.close()
        provider.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if args.mode == "open_close":
        run_open_close(args)
    elif args.mode == "gripper":
        run_gripper(args)
    elif args.mode == "vr_hand_pose":
        run_vr_hand_pose(args)
    else:
        raise AssertionError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
