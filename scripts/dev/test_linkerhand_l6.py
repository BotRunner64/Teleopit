#!/usr/bin/env python3
"""Exercise LinkerHand L6 dexterous-hand control modes."""

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
from teleopit.sim2real.dexterous_hand import (  # noqa: E402
    LinkerHandConfig,
    LinkerHandRuntime,
    SomeHandPoseRuntime,
    VR_HAND_POSE_SPEED,
)


THUMB_YAW_DEFAULT = 10
OPEN_POSE = [250, THUMB_YAW_DEFAULT, 250, 250, 250, 250]
CLOSE_POSE = [79, THUMB_YAW_DEFAULT, 0, 0, 0, 0]
DEFAULT_SPEED = [50, 50, 50, 50, 50, 50]
DEFAULT_SOMEHAND_CONFIG_PATH = "third_party/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml"
DEFAULT_LINKERHAND_SDK_ROOT = "third_party/linkerhand-python-sdk"


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
    parser = argparse.ArgumentParser(description="Test LinkerHand L6 dexterous-hand control modes")
    parser.add_argument(
        "--mode",
        choices=["open_close", "gripper", "vr_hand_pose"],
        default="open_close",
        help=(
            "open_close sends fixed poses directly; gripper reads real Pico controller "
            "grip/trigger input; vr_hand_pose reads real Pico hand pose input through somehand."
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
    parser.add_argument("--cycles", type=positive_int, default=3)
    parser.add_argument("--hold-s", type=positive_float, default=1.0)
    parser.add_argument(
        "--duration-s",
        type=positive_float,
        default=30.0,
        help="Live Pico test duration for gripper/vr_hand_pose modes.",
    )
    parser.add_argument("--rate", type=positive_float, default=30.0)
    parser.add_argument("--frame-timeout", type=positive_float, default=0.3)
    parser.add_argument("--trigger-deadzone", type=float, default=0.05)
    parser.add_argument("--deadman-threshold", type=float, default=0.5)
    parser.add_argument("--thumb-yaw-center", type=uint8, default=THUMB_YAW_DEFAULT)
    parser.add_argument("--print-input", action="store_true")
    parser.add_argument(
        "--speed",
        type=uint8,
        nargs=6,
        default=DEFAULT_SPEED,
        help="L6 speed for open_close and gripper modes. vr_hand_pose always uses max speed.",
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
    parser.add_argument("--somehand-config-path", default=DEFAULT_SOMEHAND_CONFIG_PATH)
    parser.add_argument("--somehand-sdk-root", default=DEFAULT_LINKERHAND_SDK_ROOT)
    parser.add_argument("--bridge-host", default="0.0.0.0")
    parser.add_argument("--bridge-port", type=positive_int, default=63901)
    parser.add_argument("--bridge-advertise-ip", default=None)
    parser.add_argument("--bridge-start-timeout", type=positive_float, default=10.0)
    parser.add_argument("--no-bridge-discovery", action="store_true")
    args = parser.parse_args()
    args.open_pose[1] = args.thumb_yaw_center
    args.close_pose[1] = args.thumb_yaw_center
    if args.trigger_deadzone < 0.0 or args.trigger_deadzone >= 0.5:
        raise SystemExit("--trigger-deadzone must be in [0, 0.5)")
    if args.deadman_threshold <= 0.0 or args.deadman_threshold >= 1.0:
        raise SystemExit("--deadman-threshold must be in (0, 1)")
    return args


def make_config(args: argparse.Namespace, *, mode: str) -> LinkerHandConfig:
    speed = VR_HAND_POSE_SPEED if mode == "vr_hand_pose" else args.speed
    return LinkerHandConfig(
        mode=mode,
        enabled=True,
        hand_joint="L6",
        hand_type=args.hand_type,
        left_can=args.left_can,
        right_can=args.right_can,
        modbus=args.modbus,
        rate=args.rate,
        frame_timeout=args.frame_timeout,
        trigger_deadzone=args.trigger_deadzone,
        deadman_threshold=args.deadman_threshold,
        thumb_yaw_center=args.thumb_yaw_center,
        speed=tuple(speed),
        open_pose=tuple(args.open_pose),
        close_pose=tuple(args.close_pose),
        print_input=args.print_input,
        somehand_config_path=args.somehand_config_path,
        somehand_sdk_root=args.somehand_sdk_root,
    )


def send_all(hands: dict[str, object], pose: Sequence[int], *, label: str) -> None:
    print(f"{label}: {list(pose)}", flush=True)
    for hand_type, hand in hands.items():
        print(f"  {hand_type}", flush=True)
        hand.finger_move(pose=list(pose))


def wait_runtime_idle(runtime: object, *, timeout_s: float = 2.0) -> None:
    sender = getattr(runtime, "_sender", None)
    wait_idle = getattr(sender, "wait_idle", None)
    if callable(wait_idle) and not wait_idle(timeout_s=timeout_s):
        raise RuntimeError("Timed out waiting for LinkerHand sender to become idle")


def assert_runtime_started(runtime: object) -> None:
    sender = getattr(runtime, "_sender", None)
    if not bool(getattr(sender, "started", False)):
        raise RuntimeError("LinkerHand sender failed to start; check the log above for SDK/CAN errors")


def make_pico_provider(args: argparse.Namespace) -> Pico4InputProvider:
    return Pico4InputProvider(
        timeout=args.duration_s,
        pause_button=None,
        bridge_host=args.bridge_host,
        bridge_port=args.bridge_port,
        bridge_discovery=not args.no_bridge_discovery,
        bridge_advertise_ip=args.bridge_advertise_ip,
        bridge_start_timeout=args.bridge_start_timeout,
        bridge_video=None,
        bridge_video_enabled=False,
    )


def run_live_until_done(
    runtime: LinkerHandRuntime | SomeHandPoseRuntime,
    *,
    provider: Pico4InputProvider,
    duration_s: float,
    mode_label: str,
) -> None:
    deadline = time.monotonic() + duration_s
    last_seq: int | None = None
    print(f"Running {mode_label} for {duration_s:.1f}s; press Ctrl-C to stop early.", flush=True)
    while time.monotonic() < deadline:
        now_s = time.monotonic()
        runtime.tick(active=True, now_s=now_s)
        snapshot = (
            provider.get_controller_snapshot()
            if isinstance(runtime, LinkerHandRuntime)
            else provider.get_hand_snapshot()
        )
        if snapshot is not None and snapshot.seq != last_seq:
            last_seq = snapshot.seq
            age_ms = max((now_s - snapshot.timestamp_s) * 1000.0, 0.0)
            print(f"  pico seq={snapshot.seq} age={age_ms:.1f}ms", flush=True)
        wait_runtime_idle(runtime)
        time.sleep(max(1.0 / runtime.config.rate, 0.001))


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


def run_gripper(args: argparse.Namespace) -> None:
    config = make_config(args, mode="gripper")
    provider = make_pico_provider(args)
    runtime = LinkerHandRuntime(config, provider)

    print(
        "Testing dexterous_hand.mode=gripper with real Pico controller input. "
        "Hold grip above the deadman threshold, then use trigger to close/open.",
        flush=True,
    )
    try:
        runtime.start()
        wait_runtime_idle(runtime)
        assert_runtime_started(runtime)
        run_live_until_done(runtime, provider=provider, duration_s=args.duration_s, mode_label="gripper")
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        runtime.tick(active=False)
        wait_runtime_idle(runtime)
        runtime.close()
        provider.close()


def run_vr_hand_pose(args: argparse.Namespace) -> None:
    if args.hand_type != "both":
        raise SystemExit("dexterous_hand.mode=vr_hand_pose currently requires --hand-type both")

    config = make_config(args, mode="vr_hand_pose")
    provider = make_pico_provider(args)
    runtime = SomeHandPoseRuntime(config, provider)

    print(
        "Testing dexterous_hand.mode=vr_hand_pose with real Pico hand-pose input. "
        "Enable Pico hand tracking and move both hands; start with the robot clear of contacts.",
        flush=True,
    )
    try:
        runtime.start()
        wait_runtime_idle(runtime)
        assert_runtime_started(runtime)
        run_live_until_done(runtime, provider=provider, duration_s=args.duration_s, mode_label="vr_hand_pose")
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        runtime.tick(active=False)
        wait_runtime_idle(runtime)
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
