"""Optional LinkerHand L6 control for Pico sim2real."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Protocol, Sequence

from teleopit.inputs.pico4_provider import PicoControllerSnapshot, PicoControllerState
from teleopit.runtime.common import cfg_get

logger = logging.getLogger(__name__)

THUMB_YAW_DEFAULT = 10
OPEN_POSE = [250, THUMB_YAW_DEFAULT, 250, 250, 250, 250]
CLOSE_POSE = [79, THUMB_YAW_DEFAULT, 0, 0, 0, 0]
DEFAULT_SPEED = [50, 50, 50, 50, 50, 50]
HAND_TYPES = ("left", "right")


class ControllerSnapshotProvider(Protocol):
    def get_controller_snapshot(self) -> PicoControllerSnapshot | None:
        ...


@dataclass(frozen=True)
class LinkerHandConfig:
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


def parse_linkerhand_config(cfg: Any) -> LinkerHandConfig:
    hand_cfg = cfg_get(cfg, "dexterous_hand", {}) or {}
    thumb_yaw = _uint8(cfg_get(hand_cfg, "thumb_yaw_center", THUMB_YAW_DEFAULT), "thumb_yaw_center")
    open_pose = _pose_values(cfg_get(hand_cfg, "open_pose", OPEN_POSE), "open_pose")
    close_pose = _pose_values(cfg_get(hand_cfg, "close_pose", CLOSE_POSE), "close_pose")
    open_pose[1] = thumb_yaw
    close_pose[1] = thumb_yaw

    config = LinkerHandConfig(
        enabled=bool(cfg_get(hand_cfg, "enabled", False)),
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
    )
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
                "LinkerHand SDK is required when dexterous_hand.enabled=true. "
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
            close_can = getattr(hand, "close_can", None)
            if callable(close_can):
                close_can()
            inner_hand = getattr(hand, "hand", None)
            close = getattr(inner_hand, "close", None)
            if callable(close):
                close()
        self._hands.clear()


class LinkerHandRuntime:
    """Drive LinkerHand L6 from Pico controller grip/trigger snapshots."""

    def __init__(self, config: LinkerHandConfig, provider: ControllerSnapshotProvider):
        self.config = config
        self._provider = provider
        self._sender = L6PoseSender(config)
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


class DisabledLinkerHandRuntime:
    enabled = False

    def start(self) -> None:
        pass

    def tick(self, *, active: bool, now_s: float | None = None) -> None:
        del active, now_s

    def close(self) -> None:
        pass


def build_linkerhand_runtime(cfg: Any, input_provider: Any) -> LinkerHandRuntime | DisabledLinkerHandRuntime:
    config = parse_linkerhand_config(cfg)
    if not config.enabled:
        return DisabledLinkerHandRuntime()

    input_cfg = cfg_get(cfg, "input", {}) or {}
    provider_kind = str(cfg_get(input_cfg, "provider", "")).lower()
    if provider_kind != "pico4":
        raise ValueError("dexterous_hand.enabled=true requires input.provider=pico4")
    if not callable(getattr(input_provider, "get_controller_snapshot", None)):
        raise ValueError("dexterous_hand.enabled=true requires a Pico input provider with controller snapshots")
    return LinkerHandRuntime(config, input_provider)


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
