from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from teleopit.inputs.pico4_provider import PicoControllerSnapshot, PicoControllerState, PicoHandSnapshot, PicoHandState
from teleopit.sim2real.dexterous_hand import (
    L6PoseSender,
    L6RetargetPoseMapper,
    LinkerHandRuntime,
    SomeHandPoseRuntime,
    VR_HAND_POSE_SPEED,
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


class HandSnapshotProvider:
    def __init__(self) -> None:
        self.snapshot: PicoHandSnapshot | None = None

    def get_hand_snapshot(self) -> PicoHandSnapshot | None:
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


def _hand_snapshot(
    *,
    left: PicoHandState | None = None,
    right: PicoHandState | None = None,
    timestamp_s: float = 10.0,
    seq: int = 1,
) -> PicoHandSnapshot:
    missing = PicoHandState(active=False, joints=np.zeros((26, 7), dtype=np.float64), present=False)
    return PicoHandSnapshot(
        left=left or missing,
        right=right or missing,
        timestamp_s=timestamp_s,
        seq=seq,
    )


def _hand_state(*, active: bool = True, value: float = 1.0, present: bool = True) -> PicoHandState:
    joints = np.zeros((26, 7), dtype=np.float64)
    joints[:, 0] = value
    return PicoHandState(active=active, joints=joints, present=present)


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


def test_parse_config_keeps_gripper_default_speed() -> None:
    cfg = parse_linkerhand_config(
        {
            "dexterous_hand": {
                "mode": "gripper",
                "hand_type": "both",
            }
        }
    )

    assert cfg.speed == (50, 50, 50, 50, 50, 50)


def test_parse_config_sets_vr_hand_pose_speed_to_max() -> None:
    cfg = parse_linkerhand_config(
        {
            "dexterous_hand": {
                "mode": "vr_hand_pose",
                "hand_type": "both",
                "speed": [50, 50, 50, 50, 50, 50],
            }
        }
    )

    assert cfg.speed == (255, 255, 255, 255, 255, 255)


def test_vr_hand_pose_speed_constant_is_max() -> None:
    assert tuple(VR_HAND_POSE_SPEED) == (255, 255, 255, 255, 255, 255)


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


def _install_fake_somehand(monkeypatch, *, left_qpos: list[float], right_qpos: list[float]) -> None:
    class FakeHandFrame:
        def __init__(self, *, landmarks_3d, landmarks_2d, hand_side):
            self.landmarks_3d = landmarks_3d
            self.landmarks_2d = landmarks_2d
            self.hand_side = hand_side

    class FakeBiHandFrame:
        def __init__(self, *, left=None, right=None):
            self.left = left
            self.right = right

    class FakeHandModel:
        def get_joint_name_to_qpos_index(self):
            return {
                "thumb_cmc_pitch": 0,
                "thumb_cmc_yaw": 1,
                "index_mcp_pitch": 2,
                "middle_mcp_pitch": 3,
                "ring_mcp_pitch": 4,
                "pinky_mcp_pitch": 5,
            }

    class FakeEngine:
        def __init__(self):
            self.left_engine = SimpleNamespace(
                config=SimpleNamespace(hand=SimpleNamespace(name="linkerhand_l6_left", mjcf_path="left.xml")),
                hand_model=FakeHandModel(),
            )
            self.right_engine = SimpleNamespace(
                config=SimpleNamespace(hand=SimpleNamespace(name="linkerhand_l6_right", mjcf_path="right.xml")),
                hand_model=FakeHandModel(),
            )

        @classmethod
        def from_config_path(cls, _path: str):
            return cls()

        def process(self, frame):
            return SimpleNamespace(
                left_detected=frame.left is not None,
                right_detected=frame.right is not None,
                left=SimpleNamespace(qpos=np.asarray(left_qpos, dtype=np.float64)),
                right=SimpleNamespace(qpos=np.asarray(right_qpos, dtype=np.float64)),
            )

    fake_api = SimpleNamespace(
        BiHandFrame=FakeBiHandFrame,
        BiHandRetargetingEngine=FakeEngine,
        HandFrame=FakeHandFrame,
    )
    fake_pico = SimpleNamespace(pico_hand_to_landmarks=lambda joints: np.asarray(joints, dtype=np.float64)[:21, :3])
    monkeypatch.setitem(sys.modules, "somehand.api", fake_api)
    monkeypatch.setitem(sys.modules, "somehand.pico_input", fake_pico)


def test_vr_hand_pose_runtime_holds_last_pose_when_hand_pose_disappears(monkeypatch, tmp_path) -> None:
    _install_fake_somehand(
        monkeypatch,
        left_qpos=[0.99, 0.0, 1.26, 1.26, 1.26, 1.26],
        right_qpos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    config_path = tmp_path / "linkerhand_l6_bihand.yaml"
    config_path.write_text("left: {}\nright: {}\n", encoding="utf-8")
    provider = HandSnapshotProvider()
    cfg = parse_linkerhand_config(
        {
            "input": {"provider": "pico4"},
            "dexterous_hand": {
                "mode": "vr_hand_pose",
                "hand_type": "both",
                "somehand": {"config_path": str(config_path), "sdk_root": "third_party/linkerhand-python-sdk"},
            },
        }
    )
    runtime = SomeHandPoseRuntime(cfg, provider)
    runtime.start()

    provider.snapshot = _hand_snapshot(
        left=_hand_state(active=True, value=1.0),
        right=_hand_state(active=True, value=2.0),
        timestamp_s=10.0,
    )
    runtime.tick(active=True, now_s=10.0)
    assert runtime._sender.wait_idle(timeout_s=1.0)

    assert runtime._sender._last_pose["left"] == [0, 255, 0, 0, 0, 0]
    assert runtime._sender._last_pose["right"] == [255, 255, 255, 255, 255, 255]

    provider.snapshot = _hand_snapshot(
        left=_hand_state(active=False, value=9.0),
        right=_hand_state(active=False, value=9.0, present=False),
        timestamp_s=10.1,
        seq=2,
    )
    runtime.tick(active=True, now_s=10.1)
    assert runtime._sender.wait_idle(timeout_s=1.0)

    assert runtime._sender._last_pose["left"] == [0, 255, 0, 0, 0, 0]
    assert runtime._sender._last_pose["right"] == [255, 255, 255, 255, 255, 255]

    runtime.tick(active=False, now_s=10.2)
    assert runtime._sender.wait_idle(timeout_s=1.0)
    assert runtime._sender._last_pose["left"] == list(runtime.config.open_pose)
    assert runtime._sender._last_pose["right"] == list(runtime.config.open_pose)
    runtime.close()


def test_l6_retarget_pose_mapper_uses_sdk_order_and_model_joint_names() -> None:
    class FakeHandModel:
        def get_joint_name_to_qpos_index(self):
            return {
                "thumb_pitch": 2,
                "thumb_yaw": 0,
                "index_pitch": 5,
                "middle_pitch": 1,
                "ring_pitch": 4,
                "little_pitch": 3,
            }

    qpos = np.zeros(6, dtype=np.float64)
    qpos[2] = 0.99
    qpos[0] = 0.0
    qpos[5] = 1.26
    qpos[1] = 0.0
    qpos[4] = 1.26
    qpos[3] = 0.0

    mapper = L6RetargetPoseMapper(
        FakeHandModel(),
        hand_type="right",
        sdk_root="third_party/linkerhand-python-sdk",
    )

    assert mapper.qpos_to_pose(qpos) == [0, 255, 0, 255, 0, 255]


def test_l6_retarget_pose_mapper_supports_somehand_l6_prefixed_roll_joint_names() -> None:
    class FakeHandModel:
        def get_joint_name_to_qpos_index(self):
            return {
                "lh_thumb_cmc_pitch": 8,
                "lh_thumb_cmc_roll": 9,
                "lh_thumb_dip": 10,
                "lh_index_mcp_pitch": 1,
                "lh_index_dip": 0,
                "lh_middle_mcp_pitch": 3,
                "lh_middle_dip": 2,
                "lh_ring_mcp_pitch": 5,
                "lh_ring_dip": 4,
                "lh_pinky_mcp_pitch": 7,
                "lh_pinky_dip": 6,
            }

    qpos = np.zeros(11, dtype=np.float64)
    qpos[8] = 0.99
    qpos[9] = 0.0
    qpos[1] = 1.26
    qpos[3] = 0.0
    qpos[5] = 1.26
    qpos[7] = 0.0

    mapper = L6RetargetPoseMapper(
        FakeHandModel(),
        hand_type="left",
        sdk_root="third_party/linkerhand-python-sdk",
    )

    assert mapper.qpos_to_pose(qpos) == [0, 255, 0, 255, 0, 255]


def test_l6_retarget_pose_mapper_fails_when_model_joint_mapping_is_unknown() -> None:
    class FakeHandModel:
        def get_joint_name_to_qpos_index(self):
            return {
                "thumb_pitch": 0,
                "thumb_yaw": 1,
                "index_pitch": 2,
                "middle_pitch": 3,
                "ring_pitch": 4,
            }

    with pytest.raises(ValueError, match="pinky_mcp_pitch"):
        L6RetargetPoseMapper(
            FakeHandModel(),
            hand_type="right",
            sdk_root="third_party/linkerhand-python-sdk",
        )


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
