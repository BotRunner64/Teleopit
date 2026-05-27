#!/usr/bin/env python3
"""Exercise LinkerHand L6 dexterous-hand control modes."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SDK_PATH = REPO_ROOT / "third_party" / "linkerhand-python-sdk"
if SDK_PATH.exists():
    sys.path.insert(0, str(SDK_PATH))
SOMEHAND_SRC_PATH = REPO_ROOT / "third_party" / "somehand" / "src"
if SOMEHAND_SRC_PATH.exists():
    sys.path.insert(0, str(SOMEHAND_SRC_PATH))

from teleopit.inputs.pico4_provider import (  # noqa: E402
    PicoControllerSnapshot,
    PicoControllerState,
    PicoHandSnapshot,
    PicoHandState,
)
from teleopit.sim2real.dexterous_hand import (  # noqa: E402
    LinkerHandConfig,
    LinkerHandRuntime,
    SomeHandPoseRuntime,
    trigger_to_pose,
)


THUMB_YAW_DEFAULT = 10
OPEN_POSE = [250, THUMB_YAW_DEFAULT, 250, 250, 250, 250]
CLOSE_POSE = [79, THUMB_YAW_DEFAULT, 0, 0, 0, 0]
DEFAULT_SPEED = [50, 50, 50, 50, 50, 50]
DEFAULT_SOMEHAND_CONFIG_PATH = "third_party/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml"
DEFAULT_LINKERHAND_SDK_ROOT = "third_party/linkerhand-python-sdk"

PICO_BRIDGE_TO_MEDIAPIPE = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25
]
PICO_NATIVE_TO_RH = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


class ScriptControllerProvider:
    def __init__(self) -> None:
        self.snapshot: PicoControllerSnapshot | None = None

    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        return self.snapshot


class ScriptHandProvider:
    def __init__(self) -> None:
        self.snapshot: PicoHandSnapshot | None = None

    def get_hand_snapshot(self) -> PicoHandSnapshot | None:
        return self.snapshot


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
            "open_close sends fixed poses directly; gripper exercises the sim2real Pico "
            "grip/trigger mapping; vr_hand_pose exercises the somehand Pico hand-pose path "
            "with synthetic hand landmarks."
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
    args = parser.parse_args()
    args.open_pose[1] = args.thumb_yaw_center
    args.close_pose[1] = args.thumb_yaw_center
    if args.trigger_deadzone < 0.0 or args.trigger_deadzone >= 0.5:
        raise SystemExit("--trigger-deadzone must be in [0, 0.5)")
    if args.deadman_threshold <= 0.0 or args.deadman_threshold >= 1.0:
        raise SystemExit("--deadman-threshold must be in (0, 1)")
    return args


def make_config(args: argparse.Namespace, *, mode: str) -> LinkerHandConfig:
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
        speed=tuple(args.speed),
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


def controller_snapshot(
    *,
    timestamp_s: float,
    seq: int,
    trigger: float,
    grip: float,
    config: LinkerHandConfig,
) -> PicoControllerSnapshot:
    missing = PicoControllerState(raw=False, grip=0.0, trigger=0.0, present=False)
    active = PicoControllerState(raw=True, grip=grip, trigger=trigger, present=True)
    left = active if "left" in config.selected_hand_types else missing
    right = active if "right" in config.selected_hand_types else missing
    return PicoControllerSnapshot(left=left, right=right, timestamp_s=timestamp_s, seq=seq)


def run_gripper(args: argparse.Namespace) -> None:
    config = make_config(args, mode="gripper")
    provider = ScriptControllerProvider()
    runtime = LinkerHandRuntime(config, provider)

    print("Testing dexterous_hand.mode=gripper with synthetic Pico grip/trigger snapshots", flush=True)
    try:
        runtime.start()
        wait_runtime_idle(runtime)
        assert_runtime_started(runtime)

        now_s = time.monotonic()
        print("inactive safety open", flush=True)
        runtime.tick(active=False, now_s=now_s)
        wait_runtime_idle(runtime)
        time.sleep(args.hold_s)

        seq = 1
        print("deadman released -> open", flush=True)
        now_s = time.monotonic()
        provider.snapshot = controller_snapshot(
            timestamp_s=now_s,
            seq=seq,
            trigger=1.0,
            grip=0.0,
            config=config,
        )
        runtime.tick(active=True, now_s=now_s)
        wait_runtime_idle(runtime)
        time.sleep(args.hold_s)

        sweep = [0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0]
        for cycle in range(args.cycles):
            print(f"gripper cycle {cycle + 1}/{args.cycles}", flush=True)
            for trigger in sweep:
                seq += 1
                now_s = time.monotonic()
                pose = trigger_to_pose(
                    trigger,
                    open_pose=config.open_pose,
                    close_pose=config.close_pose,
                    deadzone=config.trigger_deadzone,
                    thumb_yaw_default=config.thumb_yaw_center,
                )
                print(f"  grip=1.00 trigger={trigger:.2f} -> {pose}", flush=True)
                provider.snapshot = controller_snapshot(
                    timestamp_s=now_s,
                    seq=seq,
                    trigger=trigger,
                    grip=1.0,
                    config=config,
                )
                runtime.tick(active=True, now_s=now_s)
                wait_runtime_idle(runtime)
                time.sleep(args.hold_s)
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        runtime.tick(active=False)
        wait_runtime_idle(runtime)
        runtime.close()


def rh_to_pico_native(position: Sequence[float]) -> np.ndarray:
    return np.asarray(position, dtype=np.float64) @ PICO_NATIVE_TO_RH


def synthetic_pico_hand_joints(hand_type: str, *, curl: float) -> np.ndarray:
    curl = max(0.0, min(1.0, float(curl)))
    side_sign = -1.0 if hand_type == "left" else 1.0
    joints = np.zeros((26, 7), dtype=np.float64)

    mp_landmarks = np.zeros((21, 3), dtype=np.float64)
    mp_landmarks[0] = [0.0, 0.0, 0.0]
    finger_bases = [
        (1, side_sign * 0.035, 0.035, [0.018, 0.033, 0.046, 0.058]),
        (5, side_sign * 0.020, 0.060, [0.040, 0.070, 0.095, 0.120]),
        (9, 0.0, 0.065, [0.045, 0.080, 0.110, 0.140]),
        (13, -side_sign * 0.020, 0.060, [0.040, 0.070, 0.095, 0.120]),
        (17, -side_sign * 0.040, 0.052, [0.035, 0.060, 0.082, 0.102]),
    ]
    for base_idx, x, base_y, lengths in finger_bases:
        for offset, length in enumerate(lengths):
            bend = curl * (offset + 1) / len(lengths)
            y = base_y + length * (1.0 - 0.65 * bend)
            z = -0.055 * bend
            if base_idx == 1:
                x_pos = x + side_sign * length * 0.65
                y = 0.015 + length * (1.0 - 0.35 * bend)
            else:
                x_pos = x
            mp_landmarks[base_idx + offset] = [x_pos, y, z]

    for mp_idx, pico_idx in enumerate(PICO_BRIDGE_TO_MEDIAPIPE):
        joints[pico_idx, :3] = rh_to_pico_native(mp_landmarks[mp_idx])
    joints[0, :3] = rh_to_pico_native([0.0, 0.025, 0.0])
    return joints


def hand_snapshot(*, timestamp_s: float, seq: int, curl: float) -> PicoHandSnapshot:
    return PicoHandSnapshot(
        left=PicoHandState(active=True, joints=synthetic_pico_hand_joints("left", curl=curl), present=True),
        right=PicoHandState(active=True, joints=synthetic_pico_hand_joints("right", curl=curl), present=True),
        timestamp_s=timestamp_s,
        seq=seq,
    )


def run_vr_hand_pose(args: argparse.Namespace) -> None:
    if args.hand_type != "both":
        raise SystemExit("dexterous_hand.mode=vr_hand_pose currently requires --hand-type both")

    config = make_config(args, mode="vr_hand_pose")
    provider = ScriptHandProvider()
    runtime = SomeHandPoseRuntime(config, provider)

    print(
        "Testing dexterous_hand.mode=vr_hand_pose with synthetic Pico hand-pose snapshots. "
        "This drives poses produced by somehand; start with the robot clear of contacts.",
        flush=True,
    )
    try:
        runtime.start()
        wait_runtime_idle(runtime)
        assert_runtime_started(runtime)

        seq = 0
        curl_sweep = [0.0, 0.35, 0.7, 1.0, 0.7, 0.35, 0.0]
        for cycle in range(args.cycles):
            print(f"vr_hand_pose cycle {cycle + 1}/{args.cycles}", flush=True)
            for curl in curl_sweep:
                seq += 1
                now_s = time.monotonic()
                print(f"  synthetic curl={curl:.2f}", flush=True)
                provider.snapshot = hand_snapshot(timestamp_s=now_s, seq=seq, curl=curl)
                runtime.tick(active=True, now_s=now_s)
                wait_runtime_idle(runtime)
                time.sleep(args.hold_s)

        print("inactive mode -> configured open pose", flush=True)
        runtime.tick(active=False)
        wait_runtime_idle(runtime)
    except KeyboardInterrupt:
        print("Interrupted; opening hands before exit", flush=True)
    finally:
        runtime.tick(active=False)
        wait_runtime_idle(runtime)
        runtime.close()


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
