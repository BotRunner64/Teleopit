from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from teleopit.inputs.pico4_provider import PicoControllerSnapshot, PicoControllerState
from teleopit.sim2real.dexterous_hand import (
    L6PoseSender,
    LinkerHandRuntime,
    parse_linkerhand_config,
    trigger_to_pose,
)


class FakeInnerHand:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FakeLinkerHandApi:
    instances: list["FakeLinkerHandApi"] = []

    def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.modbus = modbus
        self.can = can
        self.hand = FakeInnerHand()
        self.close_can_calls = 0
        self.speed: list[int] | None = None
        self.poses: list[list[int]] = []
        FakeLinkerHandApi.instances.append(self)

    def set_speed(self, speed: list[int]) -> None:
        self.speed = speed

    def finger_move(self, pose: list[int]) -> None:
        self.poses.append(list(pose))

    def close_can(self) -> None:
        self.close_can_calls += 1


@pytest.fixture(autouse=True)
def fake_linkerhand_sdk(monkeypatch):
    FakeLinkerHandApi.instances = []
    fake_module = SimpleNamespace(LinkerHandApi=FakeLinkerHandApi)
    monkeypatch.setitem(sys.modules, "LinkerHand.linker_hand_api", fake_module)
    yield


class SnapshotProvider:
    def __init__(self) -> None:
        self.snapshot: PicoControllerSnapshot | None = None

    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        return self.snapshot


def _snapshot(
    *,
    left: PicoControllerState | None = None,
    right: PicoControllerState | None = None,
    timestamp_s: float = 10.0,
    seq: int = 1,
) -> PicoControllerSnapshot:
    missing = PicoControllerState(raw=False, grip=0.0, trigger=0.0, present=False)
    return PicoControllerSnapshot(
        left=left or missing,
        right=right or missing,
        timestamp_s=timestamp_s,
        seq=seq,
    )


def _runtime(provider: SnapshotProvider) -> LinkerHandRuntime:
    cfg = parse_linkerhand_config(
        {
            "dexterous_hand": {
                "enabled": True,
                "hand_type": "both",
            }
        }
    )
    runtime = LinkerHandRuntime(cfg, provider)
    runtime.start()
    return runtime


def _wait_runtime_idle(runtime: LinkerHandRuntime) -> None:
    assert runtime._sender.wait_idle(timeout_s=1.0)


def test_trigger_to_pose_applies_deadzone_and_fixed_thumb_yaw() -> None:
    pose = trigger_to_pose(
        0.5,
        open_pose=[250, 10, 250, 250, 250, 250],
        close_pose=[79, 10, 0, 0, 0, 0],
        deadzone=0.05,
        thumb_yaw_default=10,
    )

    assert pose == [164, 10, 125, 125, 125, 125]


def test_runtime_opens_when_deadman_released() -> None:
    provider = SnapshotProvider()
    runtime = _runtime(provider)
    provider.snapshot = _snapshot(
        left=PicoControllerState(raw=True, grip=0.1, trigger=1.0),
        right=PicoControllerState(raw=True, grip=0.1, trigger=1.0),
    )

    runtime.tick(active=True, now_s=10.0)
    _wait_runtime_idle(runtime)

    assert runtime._sender._last_pose["left"] == list(runtime.config.open_pose)
    assert runtime._sender._last_pose["right"] == list(runtime.config.open_pose)


def test_runtime_maps_present_controller_even_without_raw_flag() -> None:
    provider = SnapshotProvider()
    runtime = _runtime(provider)
    provider.snapshot = _snapshot(
        left=PicoControllerState(raw=False, grip=1.0, trigger=1.0, present=True),
        right=PicoControllerState(raw=False, grip=1.0, trigger=0.0, present=True),
    )

    runtime.tick(active=True, now_s=10.0)
    _wait_runtime_idle(runtime)

    assert runtime._sender._last_pose["left"] == list(runtime.config.close_pose)
    assert runtime._sender._last_pose["right"] == list(runtime.config.open_pose)


def test_runtime_maps_trigger_when_deadman_active() -> None:
    provider = SnapshotProvider()
    runtime = _runtime(provider)
    provider.snapshot = _snapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0),
        right=PicoControllerState(raw=True, grip=1.0, trigger=0.0),
    )

    runtime.tick(active=True, now_s=10.0)
    _wait_runtime_idle(runtime)

    assert runtime._sender._last_pose["left"] == list(runtime.config.close_pose)
    assert runtime._sender._last_pose["right"] == list(runtime.config.open_pose)


def test_runtime_opens_on_timeout_and_inactive_mode() -> None:
    provider = SnapshotProvider()
    runtime = _runtime(provider)
    provider.snapshot = _snapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0),
        right=PicoControllerState(raw=True, grip=1.0, trigger=1.0),
        timestamp_s=10.0,
    )

    runtime.tick(active=True, now_s=10.0)
    _wait_runtime_idle(runtime)
    assert runtime._sender._last_pose["left"] == list(runtime.config.close_pose)

    provider.snapshot = SimpleNamespace(timestamp_s=9.0, seq=2, left=None, right=None)
    runtime.tick(active=True, now_s=20.0)
    _wait_runtime_idle(runtime)
    assert runtime._sender._last_pose["left"] == list(runtime.config.open_pose)

    provider.snapshot = _snapshot(
        left=PicoControllerState(raw=True, grip=1.0, trigger=1.0),
        right=PicoControllerState(raw=True, grip=1.0, trigger=1.0),
        timestamp_s=20.1,
    )
    runtime.tick(active=True, now_s=20.1)
    _wait_runtime_idle(runtime)
    assert runtime._sender._last_pose["left"] == list(runtime.config.close_pose)

    runtime.tick(active=False, now_s=20.2)
    _wait_runtime_idle(runtime)
    assert runtime._sender._last_pose["left"] == list(runtime.config.open_pose)


def test_pose_sender_close_leaves_can_interfaces_up() -> None:
    cfg = parse_linkerhand_config(
        {
            "dexterous_hand": {
                "enabled": True,
                "hand_type": "both",
            }
        }
    )
    sender = L6PoseSender(cfg)
    sender.start()

    sender.close()

    assert [hand.close_can_calls for hand in FakeLinkerHandApi.instances] == [0, 0]
    assert [hand.hand.close_calls for hand in FakeLinkerHandApi.instances] == [1, 1]


def test_pose_sender_cleans_up_partial_start_failure(monkeypatch) -> None:
    created_hands = []

    class FailingLinkerHandApi:
        def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
            del hand_joint, modbus, can
            if hand_type == "right":
                raise RuntimeError("right hand failed")
            self.hand = FakeInnerHand()
            self.close_can_calls = 0
            created_hands.append(self)

        def set_speed(self, speed: list[int]) -> None:
            self.speed = speed

        def close_can(self) -> None:
            self.close_can_calls += 1

    fake_module = SimpleNamespace(LinkerHandApi=FailingLinkerHandApi)
    monkeypatch.setitem(sys.modules, "LinkerHand.linker_hand_api", fake_module)

    cfg = parse_linkerhand_config(
        {
            "dexterous_hand": {
                "enabled": True,
                "hand_type": "both",
            }
        }
    )
    sender = L6PoseSender(cfg)

    with pytest.raises(RuntimeError, match="right hand failed"):
        sender.start()

    assert sender.started is False
    assert sender._hands == {}
    assert len(created_hands) == 1
    assert created_hands[0].close_can_calls == 0
    assert created_hands[0].hand.close_calls == 1


def test_pose_sender_wraps_sdk_system_exit_and_cleans_up(monkeypatch) -> None:
    created_hands = []

    class ExitingLinkerHandApi:
        def __init__(self, *, hand_joint: str, hand_type: str, modbus: str, can: str) -> None:
            del hand_joint, modbus, can
            if hand_type == "right":
                raise SystemExit(1)
            self.hand = FakeInnerHand()
            self.close_can_calls = 0
            created_hands.append(self)

        def set_speed(self, speed: list[int]) -> None:
            self.speed = speed

        def close_can(self) -> None:
            self.close_can_calls += 1

    fake_module = SimpleNamespace(LinkerHandApi=ExitingLinkerHandApi)
    monkeypatch.setitem(sys.modules, "LinkerHand.linker_hand_api", fake_module)

    cfg = parse_linkerhand_config(
        {
            "dexterous_hand": {
                "enabled": True,
                "hand_type": "both",
            }
        }
    )
    sender = L6PoseSender(cfg)

    with pytest.raises(RuntimeError, match="LinkerHand SDK exited during startup"):
        sender.start()

    assert sender.started is False
    assert sender._hands == {}
    assert len(created_hands) == 1
    assert created_hands[0].close_can_calls == 0
    assert created_hands[0].hand.close_calls == 1
