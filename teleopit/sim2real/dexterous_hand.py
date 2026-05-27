"""Optional LinkerHand L6 control for Pico sim2real."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Any, Protocol, Sequence

import numpy as np

from teleopit.inputs.pico4_provider import (
    PicoControllerSnapshot,
    PicoControllerState,
    PicoHandSnapshot,
    PicoHandState,
)
from teleopit.runtime.common import cfg_get

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THUMB_YAW_DEFAULT = 10
OPEN_POSE = [250, THUMB_YAW_DEFAULT, 250, 250, 250, 250]
CLOSE_POSE = [79, THUMB_YAW_DEFAULT, 0, 0, 0, 0]
DEFAULT_SPEED = [50, 50, 50, 50, 50, 50]
HAND_TYPES = ("left", "right")
HAND_MODES = ("off", "gripper", "vr_hand_pose")
DEFAULT_SOMEHAND_CONFIG_PATH = "third_party/somehand/configs/retargeting/bihand/linkerhand_l6_bihand.yaml"
DEFAULT_LINKERHAND_SDK_ROOT = "third_party/linkerhand-python-sdk"
L6_QPOS_CHANNELS = (
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
)
L6_QPOS_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
L6_QPOS_MAX = np.array([0.99, 1.39, 1.26, 1.26, 1.26, 1.26], dtype=np.float64)


class ControllerSnapshotProvider(Protocol):
    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        ...


class HandSnapshotProvider(Protocol):
    def get_hand_snapshot(self) -> PicoHandSnapshot | None:
        ...


@dataclass(frozen=True)
class LinkerHandConfig:
    mode: str = "off"
    enabled: bool = False
    hand_joint: str = "L6"
    hand_type: str = "both"
    left_can: str = "can0"
    right_can: str = "can1"
    modbus: str = "None"
    rate: float = 30.0
    frame_timeout: float = 0.3
    trigger_deadzone: float = 0.05
    deadman_threshold: float = 0.5
    thumb_yaw_center: int = THUMB_YAW_DEFAULT
    speed: tuple[int, ...] = tuple(DEFAULT_SPEED)
    open_pose: tuple[int, ...] = tuple(OPEN_POSE)
    close_pose: tuple[int, ...] = tuple(CLOSE_POSE)
    print_input: bool = False
    somehand_config_path: str = DEFAULT_SOMEHAND_CONFIG_PATH
    somehand_sdk_root: str = DEFAULT_LINKERHAND_SDK_ROOT

    @property
    def selected_hand_types(self) -> tuple[str, ...]:
        if self.hand_type == "both":
            return HAND_TYPES
        return (self.hand_type,)


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_trigger(value: float, deadzone: float) -> float:
    value = clamp_unit(value)
    deadzone = clamp_unit(deadzone)
    if deadzone >= 0.5:
        raise ValueError(f"trigger_deadzone must be < 0.5, got {deadzone}")
    if value <= deadzone:
        return 0.0
    upper = 1.0 - deadzone
    if value >= upper:
        return 1.0
    return (value - deadzone) / (upper - deadzone)


def trigger_to_pose(
    trigger: float,
    *,
    open_pose: Sequence[int],
    close_pose: Sequence[int],
    deadzone: float,
    thumb_yaw_default: int,
) -> list[int]:
    if len(open_pose) != 6 or len(close_pose) != 6:
        raise ValueError("LinkerHand L6 open_pose and close_pose must each contain 6 values")
    alpha = normalize_trigger(trigger, deadzone)
    pose = [
        int(round(float(open_value) + alpha * (float(close_value) - float(open_value))))
        for open_value, close_value in zip(open_pose, close_pose)
    ]
    pose[1] = int(thumb_yaw_default)
    return pose


class L6RetargetPoseMapper:
    """Map somehand-retargeted L6 qpos into Teleopit's six-channel L6 SDK pose."""

    def __init__(self, hand_model: Any | None, config: LinkerHandConfig, *, hand_type: str):
        self._config = config
        self._hand_type = hand_type
        self._indices = self._resolve_indices(hand_model, hand_type=hand_type)

    def qpos_to_pose(self, qpos: Any) -> list[int]:
        values = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if self._indices is None:
            if values.shape[0] < len(L6_QPOS_CHANNELS):
                raise ValueError(
                    "somehand L6 retarget qpos is too short: "
                    f"got {values.shape[0]}, need at least {len(L6_QPOS_CHANNELS)}"
                )
            channel_values = values[:len(L6_QPOS_CHANNELS)]
        else:
            channel_values = values[self._indices]

        normalized = np.clip((channel_values - L6_QPOS_MIN) / (L6_QPOS_MAX - L6_QPOS_MIN), 0.0, 1.0)
        pose = [
            int(round(float(open_value) + float(alpha) * (float(close_value) - float(open_value))))
            for open_value, close_value, alpha in zip(
                self._config.open_pose,
                self._config.close_pose,
                normalized,
            )
        ]
        pose[1] = int(self._config.thumb_yaw_center)
        return [_uint8(value, f"somehand.{self._hand_type}.pose") for value in pose]

    @staticmethod
    def _resolve_indices(hand_model: Any | None, *, hand_type: str) -> np.ndarray | None:
        if hand_model is None:
            return None
        get_index = getattr(hand_model, "get_joint_name_to_qpos_index", None)
        if not callable(get_index):
            return None
        joint_index = get_index()
        indices: list[int] = []
        for channel in L6_QPOS_CHANNELS:
            resolved = _resolve_l6_joint_name(joint_index, channel, hand_type=hand_type)
            if resolved is None:
                return None
            indices.append(int(joint_index[resolved]))
        return np.asarray(indices, dtype=np.int64)


def parse_linkerhand_config(cfg: Any) -> LinkerHandConfig:
    hand_cfg = cfg_get(cfg, "dexterous_hand", {}) or {}
    raw_mode = cfg_get(hand_cfg, "mode", None)
    legacy_enabled = bool(cfg_get(hand_cfg, "enabled", False))
    mode = str(raw_mode if raw_mode is not None else ("gripper" if legacy_enabled else "off")).lower()
    somehand_cfg = cfg_get(hand_cfg, "somehand", {}) or {}
    thumb_yaw = _uint8(cfg_get(hand_cfg, "thumb_yaw_center", THUMB_YAW_DEFAULT), "thumb_yaw_center")
    open_pose = _pose_values(cfg_get(hand_cfg, "open_pose", OPEN_POSE), "open_pose")
    close_pose = _pose_values(cfg_get(hand_cfg, "close_pose", CLOSE_POSE), "close_pose")
    open_pose[1] = thumb_yaw
    close_pose[1] = thumb_yaw

    config = LinkerHandConfig(
        mode=mode,
        enabled=mode != "off",
        hand_joint=str(cfg_get(hand_cfg, "hand_joint", "L6")).upper(),
        hand_type=str(cfg_get(hand_cfg, "hand_type", "both")).lower(),
        left_can=str(cfg_get(hand_cfg, "left_can", "can0")),
        right_can=str(cfg_get(hand_cfg, "right_can", "can1")),
        modbus=str(cfg_get(hand_cfg, "modbus", "None")),
        rate=_positive_float(cfg_get(hand_cfg, "rate", 30.0), "rate"),
        frame_timeout=_positive_float(cfg_get(hand_cfg, "frame_timeout", 0.3), "frame_timeout"),
        trigger_deadzone=_trigger_deadzone(cfg_get(hand_cfg, "trigger_deadzone", 0.05)),
        deadman_threshold=_deadman_threshold(cfg_get(hand_cfg, "deadman_threshold", 0.5)),
        thumb_yaw_center=thumb_yaw,
        speed=tuple(_pose_values(cfg_get(hand_cfg, "speed", DEFAULT_SPEED), "speed")),
        open_pose=tuple(open_pose),
        close_pose=tuple(close_pose),
        print_input=bool(cfg_get(hand_cfg, "print_input", False)),
        somehand_config_path=str(cfg_get(somehand_cfg, "config_path", DEFAULT_SOMEHAND_CONFIG_PATH)),
        somehand_sdk_root=str(cfg_get(somehand_cfg, "sdk_root", DEFAULT_LINKERHAND_SDK_ROOT)),
    )
    if config.mode not in HAND_MODES:
        raise ValueError(f"dexterous_hand.mode must be one of {', '.join(HAND_MODES)}, got {config.mode!r}")
    if config.hand_joint != "L6":
        raise ValueError(f"dexterous_hand.hand_joint must be 'L6', got {config.hand_joint!r}")
    if config.hand_type not in ("left", "right", "both"):
        raise ValueError("dexterous_hand.hand_type must be left, right, or both")
    return config


class L6PoseSender:
    """Thin adapter around LinkerHandApi with duplicate-command suppression."""

    def __init__(self, config: LinkerHandConfig):
        self._config = config
        self._hand_types = config.selected_hand_types
        self._can_channels = {"left": config.left_can, "right": config.right_can}
        self._hands: dict[str, Any] = {}
        self._last_pose: dict[str, list[int] | None] = {
            hand_type: None for hand_type in self._hand_types
        }
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        if self._started:
            return
        try:
            from LinkerHand.linker_hand_api import LinkerHandApi
        except ImportError as exc:
            raise ImportError(
                "LinkerHand SDK is required when dexterous_hand.mode is gripper or vr_hand_pose. "
                "Run: pip install -e third_party/linkerhand-python-sdk"
            ) from exc

        try:
            for hand_type in self._hand_types:
                hand = LinkerHandApi(
                    hand_joint="L6",
                    hand_type=hand_type,
                    modbus=self._config.modbus,
                    can=self._can_channels[hand_type],
                )
                hand.set_speed(speed=list(self._config.speed))
                self._hands[hand_type] = hand
            self._started = True
        except SystemExit as exc:
            self._close_hands()
            self._started = False
            raise RuntimeError(
                "LinkerHand SDK exited during startup. Check CAN interface configuration "
                f"({', '.join(self._can_channels[hand_type] for hand_type in self._hand_types)}). "
                "Run scripts/dev/test_linkerhand_l6.py to verify the hand connection."
            ) from exc
        except Exception:
            self._close_hands()
            self._started = False
            raise
        logger.info("LinkerHand L6 runtime started | hands=%s", ",".join(self._hand_types))

    def send(self, hand_type: str, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None:
        if not self._started:
            return
        next_pose = [int(value) for value in pose]
        if not force and self._last_pose.get(hand_type) == next_pose:
            return
        del reason
        hand = self._hands.get(hand_type)
        if hand is None:
            raise RuntimeError("L6PoseSender has not been started")
        hand.finger_move(pose=next_pose)
        self._last_pose[hand_type] = next_pose

    def send_all(self, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None:
        for hand_type in self._hand_types:
            self.send(hand_type, pose, force=force, reason=reason)

    def close(self) -> None:
        if not self._started and not self._hands:
            return
        try:
            if self._started:
                self.send_all(self._config.open_pose, force=True, reason="exit")
            time.sleep(0.2)
        except Exception:
            logger.exception("Failed to send LinkerHand open pose on exit")
        self._close_hands()
        self._started = False

    def _close_hands(self) -> None:
        for hand in self._hands.values():
            inner_hand = getattr(hand, "hand", None)
            close = getattr(inner_hand, "close", None)
            if callable(close):
                close()
        self._hands.clear()


class AsyncL6PoseSender:
    """Run blocking LinkerHand SDK calls outside the robot control loop."""

    def __init__(self, config: LinkerHandConfig):
        self._config = config
        self._sync_sender = L6PoseSender(config)
        self._condition = threading.Condition()
        self._pending: dict[str, tuple[list[int], bool, str]] = {}
        self._thread: threading.Thread | None = None
        self._running = False
        self._stopping = False
        self._busy = False
        self._failed = False

    @property
    def started(self) -> bool:
        return self._running and not self._failed

    @property
    def _last_pose(self) -> dict[str, list[int] | None]:
        return self._sync_sender._last_pose

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._stopping = False
            self._failed = False
            self._busy = True
            self._thread = threading.Thread(
                target=self._run,
                name="linkerhand-l6-sender",
                daemon=True,
            )
            self._thread.start()

    def send(self, hand_type: str, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None:
        next_pose = [int(value) for value in pose]
        if not force and self._sync_sender._last_pose.get(hand_type) == next_pose:
            return
        with self._condition:
            if not self._running or self._failed or self._stopping:
                return
            self._pending[hand_type] = (next_pose, force, reason)
            self._condition.notify_all()

    def send_all(self, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None:
        for hand_type in self._config.selected_hand_types:
            self.send(hand_type, pose, force=force, reason=reason)

    def close(self) -> None:
        thread: threading.Thread | None
        with self._condition:
            if not self._running:
                return
            if not self._failed:
                for hand_type in self._config.selected_hand_types:
                    self._pending[hand_type] = (list(self._config.open_pose), True, "exit")
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if thread.is_alive():
                logger.warning("LinkerHand L6 worker did not stop within timeout")

    def wait_idle(self, timeout_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._busy or self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def _run(self) -> None:
        try:
            self._sync_sender.start()
            self._sync_sender.send_all(self._config.open_pose, force=True, reason="startup")
            while True:
                commands = self._take_commands()
                if not commands:
                    break
                try:
                    for hand_type, pose, force, reason in commands:
                        self._sync_sender.send(hand_type, pose, force=force, reason=reason)
                finally:
                    with self._condition:
                        self._busy = False
                        self._condition.notify_all()
        except Exception:
            logger.exception("LinkerHand L6 worker failed; hand control is disabled")
            with self._condition:
                self._failed = True
                self._pending.clear()
                self._busy = False
                self._condition.notify_all()
        finally:
            try:
                self._sync_sender.close()
            except Exception:
                logger.exception("Failed to close LinkerHand L6 worker cleanly")
            with self._condition:
                self._running = False
                self._busy = False
                self._condition.notify_all()

    def _take_commands(self) -> list[tuple[str, list[int], bool, str]]:
        with self._condition:
            while not self._pending and not self._stopping:
                self._busy = False
                self._condition.notify_all()
                self._condition.wait()
            if not self._pending and self._stopping:
                return []
            self._busy = True
            commands = [
                (hand_type, pose, force, reason)
                for hand_type, (pose, force, reason) in self._pending.items()
            ]
            self._pending.clear()
            return commands


class LinkerHandRuntime:
    """Drive LinkerHand L6 from Pico controller grip/trigger snapshots."""

    def __init__(self, config: LinkerHandConfig, provider: ControllerSnapshotProvider):
        self.config = config
        self._provider = provider
        self._sender = AsyncL6PoseSender(config)
        self._interval_s = 1.0 / config.rate
        self._next_tick_s = 0.0
        self._active = False
        self._last_status: dict[str, str] = {hand_type: "" for hand_type in config.selected_hand_types}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        if not self.enabled:
            return
        self._sender.start()
        self._sender.send_all(self.config.open_pose, force=True, reason="startup")

    def tick(self, *, active: bool, now_s: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.monotonic() if now_s is None else float(now_s)
        if not active:
            self._deactivate(reason="inactive")
            return
        if not self._active:
            self._active = True
            self._next_tick_s = 0.0
        if now < self._next_tick_s:
            return
        self._next_tick_s = now + self._interval_s

        snapshot = self._provider.get_controller_snapshot()
        if snapshot is None or now - snapshot.timestamp_s > self.config.frame_timeout:
            self._open_all(reason="timeout")
            return

        for hand_type in self.config.selected_hand_types:
            state = getattr(snapshot, hand_type)
            self._tick_hand(hand_type, state, snapshot.seq)

    def close(self) -> None:
        self._deactivate(reason="shutdown")
        self._sender.close()

    def _tick_hand(self, hand_type: str, state: PicoControllerState, seq: int) -> None:
        if not state.present:
            self._set_status(hand_type, "missing", f"{hand_type} controller missing; opening hand")
            self._sender.send(hand_type, self.config.open_pose, reason="missing-controller")
            return

        grip = clamp_unit(state.grip)
        trigger = clamp_unit(state.trigger)
        if grip < self.config.deadman_threshold:
            self._set_status(hand_type, "deadman", f"{hand_type} deadman released; opening hand")
            self._sender.send(hand_type, self.config.open_pose, reason="deadman-released")
            return

        self._set_status(hand_type, "enabled", f"{hand_type} controller active")
        if self.config.print_input:
            logger.info(
                "LinkerHand input | seq=%d hand=%s grip=%.3f trigger=%.3f",
                seq,
                hand_type,
                grip,
                trigger,
            )
        pose = trigger_to_pose(
            trigger,
            open_pose=self.config.open_pose,
            close_pose=self.config.close_pose,
            deadzone=self.config.trigger_deadzone,
            thumb_yaw_default=self.config.thumb_yaw_center,
        )
        self._sender.send(hand_type, pose, reason="controller")

    def _deactivate(self, *, reason: str) -> None:
        if self._active:
            self._open_all(reason=reason, force=True)
        self._active = False

    def _open_all(self, *, reason: str, force: bool = False) -> None:
        self._sender.send_all(self.config.open_pose, force=force, reason=reason)

    def _set_status(self, hand_type: str, status: str, message: str) -> None:
        if self._last_status.get(hand_type) == status:
            return
        self._last_status[hand_type] = status
        logger.info("LinkerHand L6: %s", message)


class SomeHandPoseRuntime:
    """Drive LinkerHand L6 from Pico hand-pose snapshots through somehand."""

    def __init__(self, config: LinkerHandConfig, provider: HandSnapshotProvider):
        self.config = config
        self._provider = provider
        self._sender = AsyncL6PoseSender(config)
        self._interval_s = 1.0 / config.rate
        self._next_tick_s = 0.0
        self._active = False
        self._last_status: dict[str, str] = {hand_type: "" for hand_type in config.selected_hand_types}
        self._engine: Any | None = None
        self._hand_frame_cls: Any | None = None
        self._bihand_frame_cls: Any | None = None
        self._pico_hand_to_landmarks: Any | None = None
        self._pose_mappers: dict[str, L6RetargetPoseMapper] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        if not self.enabled:
            return
        self._load_somehand()
        self._sender.start()
        self._sender.send_all(self.config.open_pose, force=True, reason="startup")

    def tick(self, *, active: bool, now_s: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.monotonic() if now_s is None else float(now_s)
        if not active:
            self._deactivate(reason="inactive")
            return
        if not self._active:
            self._active = True
            self._next_tick_s = 0.0
        if now < self._next_tick_s:
            return
        self._next_tick_s = now + self._interval_s

        snapshot = self._provider.get_hand_snapshot()
        if snapshot is None:
            self._set_status("both", "missing", "Pico hand pose missing; holding last hand command")
            return
        if now - snapshot.timestamp_s > self.config.frame_timeout:
            self._set_status("both", "timeout", "Pico hand pose timed out; holding last hand command")
            return

        self._tick_snapshot(snapshot)

    def close(self) -> None:
        self._deactivate(reason="shutdown")
        self._sender.close()

    def _tick_snapshot(self, snapshot: PicoHandSnapshot) -> None:
        left_frame = self._make_hand_frame("left", snapshot.left) if "left" in self.config.selected_hand_types else None
        right_frame = self._make_hand_frame("right", snapshot.right) if "right" in self.config.selected_hand_types else None
        if left_frame is None and right_frame is None:
            return

        result = self._engine.process(self._bihand_frame_cls(left=left_frame, right=right_frame))
        for hand_type, detected, step in (
            ("left", result.left_detected, result.left),
            ("right", result.right_detected, result.right),
        ):
            if hand_type not in self.config.selected_hand_types or not detected:
                continue
            pose = self._pose_mappers[hand_type].qpos_to_pose(step.qpos)
            self._sender.send(hand_type, pose, reason="vr-hand-pose")

    def _make_hand_frame(self, hand_type: str, state: PicoHandState) -> Any | None:
        if not state.present:
            self._set_status(hand_type, "missing", f"{hand_type} hand pose missing; holding last hand command")
            return None
        if not state.active:
            self._set_status(hand_type, "inactive", f"{hand_type} hand pose inactive; holding last hand command")
            return None
        self._set_status(hand_type, "enabled", f"{hand_type} hand pose active")
        landmarks = self._pico_hand_to_landmarks(state.joints)
        return self._hand_frame_cls(landmarks_3d=landmarks, landmarks_2d=None, hand_side=hand_type)

    def _deactivate(self, *, reason: str) -> None:
        if self._active:
            self._sender.send_all(self.config.open_pose, force=True, reason=reason)
        self._active = False

    def _load_somehand(self) -> None:
        try:
            from somehand.api import BiHandFrame, BiHandRetargetingEngine, HandFrame
            from somehand.pico_input import pico_hand_to_landmarks
        except ImportError as exc:
            raise ImportError(
                "somehand is required when dexterous_hand.mode=vr_hand_pose. "
                "Install it with: pip install -e '.[dexhand]'"
            ) from exc

        config_path = _resolve_project_path(self.config.somehand_config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                "somehand bi-hand config not found: "
                f"{config_path}. Initialize the submodule and download assets with "
                "scripts/setup/download_somehand_l6_assets.sh"
            )
        self._engine = BiHandRetargetingEngine.from_config_path(str(config_path))
        self._hand_frame_cls = HandFrame
        self._bihand_frame_cls = BiHandFrame
        self._pico_hand_to_landmarks = pico_hand_to_landmarks

        # somehand owns hand-pose retargeting; Teleopit owns the LinkerHand L6 command mapping.
        self._pose_mappers = {}
        for hand_type, engine in (("left", self._engine.left_engine), ("right", self._engine.right_engine)):
            if hand_type not in self.config.selected_hand_types:
                continue
            self._pose_mappers[hand_type] = L6RetargetPoseMapper(
                getattr(engine, "hand_model", None),
                self.config,
                hand_type=hand_type,
            )
        logger.info("somehand LinkerHand L6 runtime started | hands=%s", ",".join(self.config.selected_hand_types))

    def _set_status(self, hand_type: str, status: str, message: str) -> None:
        key = hand_type
        if self._last_status.get(key) == status:
            return
        self._last_status[key] = status
        logger.info("somehand LinkerHand L6: %s", message)


class DisabledLinkerHandRuntime:
    enabled = False

    def start(self) -> None:
        pass

    def tick(self, *, active: bool, now_s: float | None = None) -> None:
        del active, now_s

    def close(self) -> None:
        pass


def build_linkerhand_runtime(cfg: Any, input_provider: Any) -> LinkerHandRuntime | SomeHandPoseRuntime | DisabledLinkerHandRuntime:
    config = parse_linkerhand_config(cfg)
    if not config.enabled:
        return DisabledLinkerHandRuntime()

    input_cfg = cfg_get(cfg, "input", {}) or {}
    provider_kind = str(cfg_get(input_cfg, "provider", "")).lower()
    if provider_kind != "pico4":
        raise ValueError("dexterous_hand.mode requires input.provider=pico4")
    if config.mode == "gripper":
        if not callable(getattr(input_provider, "get_controller_snapshot", None)):
            raise ValueError("dexterous_hand.mode=gripper requires a Pico input provider with controller snapshots")
        return LinkerHandRuntime(config, input_provider)
    if config.mode == "vr_hand_pose":
        if config.hand_type != "both":
            raise ValueError("dexterous_hand.mode=vr_hand_pose currently requires dexterous_hand.hand_type=both")
        if not callable(getattr(input_provider, "get_hand_snapshot", None)):
            raise ValueError("dexterous_hand.mode=vr_hand_pose requires a Pico input provider with hand snapshots")
        return SomeHandPoseRuntime(config, input_provider)
    raise ValueError(f"Unsupported dexterous_hand.mode={config.mode!r}")


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _resolve_l6_joint_name(joint_index: dict[str, int], semantic_name: str, *, hand_type: str) -> str | None:
    candidates = (
        semantic_name,
        f"{hand_type}_{semantic_name}",
        f"{hand_type[0]}_{semantic_name}",
        f"{'lh' if hand_type == 'left' else 'rh'}_{semantic_name}",
    )
    for candidate in candidates:
        if candidate in joint_index:
            return candidate
    suffix = f"_{semantic_name}"
    for name in joint_index:
        if name.endswith(suffix):
            return name
    return None


def _positive_float(value: object, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"dexterous_hand.{field_name} must be > 0, got {value!r}")
    return parsed


def _uint8(value: object, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 255:
        raise ValueError(f"dexterous_hand.{field_name} must be in range 0-255, got {value!r}")
    return parsed


def _pose_values(value: object, field_name: str) -> list[int]:
    try:
        parsed = [_uint8(item, field_name) for item in value]  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(f"dexterous_hand.{field_name} must be a sequence of 6 uint8 values") from exc
    if len(parsed) != 6:
        raise ValueError(f"dexterous_hand.{field_name} must contain 6 values, got {len(parsed)}")
    return parsed


def _trigger_deadzone(value: object) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed >= 0.5:
        raise ValueError(f"dexterous_hand.trigger_deadzone must be in [0, 0.5), got {value!r}")
    return parsed


def _deadman_threshold(value: object) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError(f"dexterous_hand.deadman_threshold must be in (0, 1), got {value!r}")
    return parsed
