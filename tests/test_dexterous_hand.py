from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from teleopit.inputs.pico4_provider import PicoControllerSnapshot, PicoControllerState
from teleopit.sim2real.hands.linkerhand_l6 import (
    GripperMapper,
    LinkerHandL6Device,
    SomehandL6Mapper,
    parse_linkerhand_l6_config,
    trigger_to_pose,
)
from teleopit.sim2real.hands.base import HandPoseCommand
from teleopit.sim2real.hands.linkerhand_o6 import (
    CLOSE_POSE as O6_CLOSE_POSE,
    LinkerHandO6Device,
    parse_linkerhand_o6_config,
)
from teleopit.sim2real.hands.pico_landmarks import pico_hand_to_landmarks
from teleopit.sim2real.hands.worker import HandRuntime


class FakeInnerHand:
    def __init__(self) -> None:
        self.close_calls = 0
        self.close_can_interface_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def close_can_interface(self) -> None:
        self.close_can_interface_calls += 1


class FakeLinkerHandApi:
    instances: list["FakeLinkerHandApi"] = []

    def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.modbus = modbus
        self.can = can
        self.hand = FakeInnerHand()
        self.speed: list[int] | None = None
        self.poses: list[list[int]] = []
        self.close_can_calls = 0
        FakeLinkerHandApi.instances.append(self)

    def set_speed(self, speed: list[int]) -> None:
        self.speed = list(speed)

    def finger_move(self, pose: list[int]) -> None:
        self.poses.append(list(pose))

    def close_can(self) -> None:
        self.close_can_calls += 1


def _cfg(mode: str = "gripper") -> dict[str, object]:
    return {
        "input": {"provider": "pico4"},
        "hands": {
            "enabled": True,
            "driver": "linkerhand_l6",
            "mode": mode,
            "sides": ["left", "right"],
            "rate_hz": 30.0,
            "frame_timeout_s": 0.3,
            "linkerhand_l6": {
                "left_can": "can0",
                "right_can": "can1",
                "modbus": "None",
                "trigger_deadzone": 0.05,
                "deadman_threshold": 0.5,
                "open_pose": [250, 10, 250, 250, 250, 250],
                "close_pose": [79, 10, 0, 0, 0, 0],
            },
            "somehand": {
                "rate_hz": 60.0,
                "max_iterations": 12,
                "temporal_filter_alpha": 1.0,
                "output_alpha": 1.0,
            },
        },
    }


def _o6_cfg(mode: str = "gripper") -> dict[str, object]:
    return {
        "input": {"provider": "pico4"},
        "hands": {
            "enabled": True,
            "driver": "linkerhand_o6",
            "mode": mode,
            "sides": ["left", "right"],
            "rate_hz": 30.0,
            "frame_timeout_s": 0.3,
            "linkerhand_o6": {
                "left_can": "can0",
                "right_can": "can1",
                "modbus": "None",
                "trigger_deadzone": 0.05,
                "deadman_threshold": 0.5,
            },
        },
    }


def test_pico_hand_to_landmarks_uses_teleopit_adapter() -> None:
    joints = np.zeros((26, 7), dtype=np.float64)
    joints[:, 0] = np.arange(26)
    joints[:, 1] = np.arange(26) + 100
    joints[:, 2] = np.arange(26) + 200

    landmarks = pico_hand_to_landmarks(joints)

    assert landmarks.shape == (21, 3)
    np.testing.assert_allclose(landmarks[0], [1.0, -201.0, 101.0])
    np.testing.assert_allclose(landmarks[-1], [25.0, -225.0, 125.0])


def test_gripper_mapper_maps_trigger_and_deadman() -> None:
    cfg = parse_linkerhand_l6_config(_cfg())
    mapper = GripperMapper(cfg)
    snapshot = PicoControllerSnapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=True, grip=0.1, trigger=1.0, present=True),
        timestamp_s=10.0,
        seq=1,
    )

    commands = mapper.map(controller_snapshot=snapshot, hand_snapshot=None, active=True, now_s=10.0)

    assert commands[0].side == "left"
    assert commands[0].pose == cfg.close_pose
    assert commands[1].side == "right"
    assert commands[1].pose == cfg.open_pose


def test_hand_mappers_force_open_once_when_inactive() -> None:
    cfg = parse_linkerhand_l6_config(_cfg())
    snapshot = PicoControllerSnapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        timestamp_s=10.0,
        seq=1,
    )

    gripper = GripperMapper(cfg)
    assert gripper.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=9.0) == ()
    assert gripper.map(controller_snapshot=snapshot, hand_snapshot=None, active=True, now_s=10.0)
    first_inactive = gripper.map(controller_snapshot=snapshot, hand_snapshot=None, active=False, now_s=10.1)
    assert [command.force for command in first_inactive] == [True, True]
    assert gripper.map(controller_snapshot=snapshot, hand_snapshot=None, active=False, now_s=10.2) == ()

    somehand = SomehandL6Mapper(cfg)
    assert somehand.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=9.0) == ()
    somehand._active = True
    first_inactive = somehand.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=10.0)
    assert [command.force for command in first_inactive] == [True, True]
    assert somehand.map(controller_snapshot=None, hand_snapshot=None, active=False, now_s=10.1) == ()


def test_trigger_to_pose_applies_deadzone_and_fixed_thumb_yaw() -> None:
    assert trigger_to_pose(
        0.5,
        open_pose=[250, 10, 250, 250, 250, 250],
        close_pose=[79, 10, 0, 0, 0, 0],
        deadzone=0.05,
        thumb_yaw_default=10,
    ) == [164, 10, 125, 125, 125, 125]


def test_trigger_to_pose_can_interpolate_thumb_yaw_for_o6() -> None:
    assert trigger_to_pose(
        1.0,
        open_pose=[250, 250, 250, 250, 250, 250],
        close_pose=[86, 73, 118, 111, 110, 111],
        deadzone=0.05,
    ) == list(O6_CLOSE_POSE)


def test_linkerhand_l6_device_starts_sdk(monkeypatch) -> None:
    FakeLinkerHandApi.instances = []
    monkeypatch.setitem(
        sys.modules,
        "LinkerHand.linker_hand_api",
        SimpleNamespace(LinkerHandApi=FakeLinkerHandApi),
    )
    cfg = parse_linkerhand_l6_config(_cfg())
    device = LinkerHandL6Device(cfg)

    device.connect()
    device.send_pose("left", cfg.close_pose)
    device.close()

    assert [hand.can for hand in FakeLinkerHandApi.instances] == ["can0", "can1"]
    assert FakeLinkerHandApi.instances[0].speed == [50, 50, 50, 50, 50, 50]
    assert FakeLinkerHandApi.instances[0].poses[-2] == list(cfg.close_pose)
    assert [hand.hand.close_calls for hand in FakeLinkerHandApi.instances] == [1, 1]


def test_linkerhand_o6_gripper_defaults_to_reference_grasp_pose() -> None:
    cfg = parse_linkerhand_o6_config(_o6_cfg())
    mapper = GripperMapper(cfg)
    snapshot = PicoControllerSnapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=True, grip=0.1, trigger=1.0, present=True),
        timestamp_s=10.0,
        seq=1,
    )

    commands = mapper.map(controller_snapshot=snapshot, hand_snapshot=None, active=True, now_s=10.0)

    assert cfg.close_pose == O6_CLOSE_POSE
    assert commands[0].pose == O6_CLOSE_POSE
    assert commands[1].pose == cfg.open_pose


def test_linkerhand_o6_device_starts_sdk(monkeypatch) -> None:
    FakeLinkerHandApi.instances = []
    monkeypatch.setitem(
        sys.modules,
        "LinkerHand.linker_hand_api",
        SimpleNamespace(LinkerHandApi=FakeLinkerHandApi),
    )
    cfg = parse_linkerhand_o6_config(_o6_cfg())
    device = LinkerHandO6Device(cfg)

    device.connect()
    device.send_pose("left", cfg.close_pose)
    device.close()

    assert [hand.hand_joint for hand in FakeLinkerHandApi.instances] == ["O6", "O6"]
    assert [hand.can for hand in FakeLinkerHandApi.instances] == ["can0", "can1"]
    assert FakeLinkerHandApi.instances[0].speed == [255, 255, 255, 255, 255, 255]
    assert FakeLinkerHandApi.instances[0].poses[-2] == list(O6_CLOSE_POSE)
    assert [hand.close_can_calls for hand in FakeLinkerHandApi.instances] == [0, 0]
    assert [hand.hand.close_can_interface_calls for hand in FakeLinkerHandApi.instances] == [1, 1]
    assert [hand.hand.close_calls for hand in FakeLinkerHandApi.instances] == [0, 0]


def test_linkerhand_o6_rejects_vr_hand_pose() -> None:
    with pytest.raises(ValueError, match="supports only hands.mode=gripper"):
        parse_linkerhand_o6_config(_o6_cfg(mode="vr_hand_pose"))


def test_hand_runtime_closes_device_when_mapper_start_fails() -> None:
    calls: list[str] = []

    class FakeDevice:
        def connect(self) -> None:
            calls.append("connect")

        def send_pose(self, *args, **kwargs) -> None:
            raise AssertionError("send_pose should not be called")

        def open_all(self, *args, **kwargs) -> None:
            calls.append("open_all")

        def close(self) -> None:
            calls.append("close")

    class FailingMapper:
        def start(self) -> None:
            calls.append("mapper_start")
            raise RuntimeError("mapper failed")

        def map(self, *args, **kwargs):
            return ()

        def close(self) -> None:
            calls.append("mapper_close")

    runtime = HandRuntime(FakeDevice(), FailingMapper())

    with pytest.raises(RuntimeError, match="mapper failed"):
        runtime.start()

    assert calls == ["connect", "mapper_start", "close"]


def test_hand_runtime_reports_actual_open_commands() -> None:
    calls: list[tuple[str, object, object]] = []
    open_commands = (
        HandPoseCommand("left", (250, 10, 250, 250, 250, 250), True, "open"),
        HandPoseCommand("right", (250, 10, 250, 250, 250, 250), True, "open"),
    )

    class FakeDevice:
        def connect(self) -> None:
            calls.append(("connect", None, None))

        def send_pose(self, side, pose, *, force=False, reason="") -> None:
            calls.append((side, tuple(pose), reason))

        def open_all(self, *, force=False, reason="") -> None:
            calls.append(("open_all", force, reason))

        def close(self) -> None:
            calls.append(("close", None, None))

    class Mapper:
        def __init__(self) -> None:
            self.fail = False

        def start(self) -> None:
            calls.append(("mapper_start", None, None))

        def map(self, *args, **kwargs):
            if self.fail:
                raise RuntimeError("tick failed")
            return (HandPoseCommand("left", (1, 2, 3, 4, 5, 6), False, "mapped"),)

        def close(self) -> None:
            calls.append(("mapper_close", None, None))

    mapper = Mapper()
    runtime = HandRuntime(FakeDevice(), mapper, open_commands=open_commands)

    startup = runtime.start()
    ticked = runtime.tick(controller_snapshot=None, hand_snapshot=None, active=True, now_s=1.0)
    mapper.fail = True
    failure = runtime.tick(controller_snapshot=None, hand_snapshot=None, active=True, now_s=2.0)
    shutdown = runtime.close()

    assert [command.reason for command in startup] == ["startup", "startup"]
    assert ticked[0].pose == (1, 2, 3, 4, 5, 6)
    assert [command.reason for command in failure] == ["failure", "failure"]
    assert [command.reason for command in shutdown] == ["shutdown", "shutdown"]
    assert ("open_all", True, "failure") in calls
    assert ("close", None, None) in calls


def test_linkerhand_l6_device_wraps_sdk_system_exit_and_cleans_up(monkeypatch) -> None:
    created_hands = []

    class ExitingLinkerHandApi:
        def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
            del hand_joint, modbus, can
            if hand_type == "right":
                raise SystemExit(1)
            self.hand = FakeInnerHand()
            created_hands.append(self)

        def set_speed(self, speed: list[int]) -> None:
            self.speed = list(speed)

        def finger_move(self, pose: list[int]) -> None:
            self.pose = list(pose)

    monkeypatch.setitem(
        sys.modules,
        "LinkerHand.linker_hand_api",
        SimpleNamespace(LinkerHandApi=ExitingLinkerHandApi),
    )
    cfg = parse_linkerhand_l6_config(_cfg())
    device = LinkerHandL6Device(cfg)

    with pytest.raises(RuntimeError, match="LinkerHand SDK exited during startup"):
        device.connect()

    assert len(created_hands) == 1
    assert created_hands[0].hand.close_calls == 1
