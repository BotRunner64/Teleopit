from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from teleopit.runtime.common import cfg_get
from teleopit.sim2real.hands.base import HAND_SIDES, HandDevice, HandInputMapper
from teleopit.sim2real.hands.linkerhand_l6 import GripperMapper

logger = logging.getLogger(__name__)

OPEN_POSE = (250, 250, 250, 250, 250, 250)
CLOSE_POSE = (86, 73, 118, 111, 110, 111)
DEFAULT_SPEED = (255, 255, 255, 255, 255, 255)


@dataclass(frozen=True)
class LinkerHandO6Config:
    mode: str
    sides: tuple[str, ...]
    left_can: str
    right_can: str
    modbus: str
    rate_hz: float
    frame_timeout_s: float
    trigger_deadzone: float
    deadman_threshold: float
    speed: tuple[int, ...]
    open_pose: tuple[int, ...]
    close_pose: tuple[int, ...]
    fixed_thumb_yaw: int | None
    print_input: bool


def parse_linkerhand_o6_config(cfg: Any) -> LinkerHandO6Config:
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    o6_cfg = cfg_get(hands_cfg, "linkerhand_o6", {}) or {}
    mode = str(cfg_get(hands_cfg, "mode", "gripper")).strip().lower()
    if mode != "gripper":
        raise ValueError(f"hands.driver=linkerhand_o6 supports only hands.mode=gripper, got {mode!r}")
    sides = tuple(str(side).strip().lower() for side in cfg_get(hands_cfg, "sides", HAND_SIDES))
    if not sides or any(side not in HAND_SIDES for side in sides):
        raise ValueError("hands.sides must contain left, right, or both sides")
    return LinkerHandO6Config(
        mode=mode,
        sides=sides,
        left_can=str(cfg_get(o6_cfg, "left_can", "can0")),
        right_can=str(cfg_get(o6_cfg, "right_can", "can1")),
        modbus=str(cfg_get(o6_cfg, "modbus", "None")),
        rate_hz=_positive_float(cfg_get(hands_cfg, "rate_hz", cfg_get(o6_cfg, "rate_hz", 30.0)), "rate_hz"),
        frame_timeout_s=_positive_float(cfg_get(hands_cfg, "frame_timeout_s", 0.3), "frame_timeout_s"),
        trigger_deadzone=_deadzone(cfg_get(o6_cfg, "trigger_deadzone", 0.05)),
        deadman_threshold=_threshold(cfg_get(o6_cfg, "deadman_threshold", 0.5)),
        speed=tuple(_pose_values(cfg_get(o6_cfg, "speed", DEFAULT_SPEED), "speed")),
        open_pose=tuple(_pose_values(cfg_get(o6_cfg, "open_pose", OPEN_POSE), "open_pose")),
        close_pose=tuple(_pose_values(cfg_get(o6_cfg, "close_pose", CLOSE_POSE), "close_pose")),
        fixed_thumb_yaw=None,
        print_input=bool(cfg_get(o6_cfg, "print_input", False)),
    )


class LinkerHandO6Device(HandDevice):
    def __init__(self, config: LinkerHandO6Config):
        self.config = config
        self._hands: dict[str, Any] = {}
        self._last_pose: dict[str, tuple[int, ...] | None] = {side: None for side in config.sides}

    def connect(self) -> None:
        try:
            from LinkerHand.linker_hand_api import LinkerHandApi
        except ImportError as exc:
            raise ImportError(
                "LinkerHand SDK is required for hands.driver=linkerhand_o6. "
                "Install it with: pip install -e third_party/linkerhand-python-sdk"
            ) from exc
        try:
            for side in self.config.sides:
                hand = LinkerHandApi(
                    hand_joint="O6",
                    hand_type=side,
                    modbus=self.config.modbus,
                    can=self.config.left_can if side == "left" else self.config.right_can,
                )
                hand.set_speed(speed=list(self.config.speed))
                self._hands[side] = hand
        except (Exception, SystemExit) as exc:
            self.close()
            if isinstance(exc, SystemExit):
                raise RuntimeError("LinkerHand SDK exited during startup") from exc
            raise
        self.open_all(force=True, reason="startup")

    def send_pose(self, side: str, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None:
        del reason
        next_pose = tuple(_uint8(value, f"{side}.pose") for value in pose)
        if not force and self._last_pose.get(side) == next_pose:
            return
        hand = self._hands.get(side)
        if hand is None:
            return
        hand.finger_move(pose=list(next_pose))
        self._last_pose[side] = next_pose

    def open_all(self, *, force: bool = False, reason: str = "") -> None:
        for side in self.config.sides:
            self.send_pose(side, self.config.open_pose, force=force, reason=reason)

    def close(self) -> None:
        try:
            self.open_all(force=True, reason="shutdown")
        except Exception:
            logger.exception("Failed to open LinkerHand O6 on shutdown")
        for hand in self._hands.values():
            inner = getattr(hand, "hand", None)
            close_backend = getattr(inner, "close_can_interface", None)
            if callable(close_backend):
                close_backend()
                continue
            close = getattr(inner, "close", None)
            if callable(close):
                close()
        self._hands.clear()


def build_linkerhand_o6(cfg: Any) -> tuple[HandDevice, HandInputMapper]:
    config = parse_linkerhand_o6_config(cfg)
    return LinkerHandO6Device(config), GripperMapper(config)


def _uint8(value: object, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > 255:
        raise ValueError(f"hands.linkerhand_o6.{field_name} must be in 0-255, got {value!r}")
    return parsed


def _pose_values(value: object, field_name: str) -> list[int]:
    parsed = [_uint8(item, field_name) for item in value]  # type: ignore[union-attr]
    if len(parsed) != 6:
        raise ValueError(f"hands.linkerhand_o6.{field_name} must contain 6 values")
    return parsed


def _positive_float(value: object, field_name: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"hands.{field_name} must be > 0")
    return parsed


def _deadzone(value: object) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed >= 0.5:
        raise ValueError("hands.linkerhand_o6.trigger_deadzone must be in [0, 0.5)")
    return parsed


def _threshold(value: object) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed >= 1.0:
        raise ValueError("hands.linkerhand_o6.deadman_threshold must be in (0, 1)")
    return parsed
