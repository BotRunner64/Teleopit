from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from teleopit.runtime.common import cfg_get
from teleopit.sim2real.hands.base import HandDevice, HandInputMapper, HandPoseCommand
from teleopit.sim2real.hands.linkerhand_l6 import build_linkerhand_l6
from teleopit.sim2real.hands.linkerhand_o6 import build_linkerhand_o6

logger = logging.getLogger(__name__)


class HandRuntime:
    def __init__(
        self,
        device: HandDevice,
        mapper: HandInputMapper,
        *,
        open_commands: Sequence[HandPoseCommand] = (),
    ):
        self._device = device
        self._mapper = mapper
        self.enabled = True
        self._failed = False
        self._open_commands = tuple(open_commands)

    def start(self) -> tuple[HandPoseCommand, ...]:
        try:
            self._device.connect()
            self._mapper.start()
            return self._open_pose_commands("startup")
        except Exception:
            try:
                self._device.close()
            finally:
                raise

    def tick(
        self,
        *,
        controller_snapshot: object | None,
        hand_snapshot: object | None,
        active: bool,
        now_s: float | None = None,
    ) -> tuple[HandPoseCommand, ...]:
        if self._failed:
            return ()
        now = time.monotonic() if now_s is None else float(now_s)
        try:
            commands = self._mapper.map(
                controller_snapshot=controller_snapshot,
                hand_snapshot=hand_snapshot,
                active=active,
                now_s=now,
            )
            sent: list[HandPoseCommand] = []
            for command in commands:
                self._device.send_pose(command.side, command.pose, force=command.force, reason=command.reason)
                sent.append(command)
            return tuple(sent)
        except Exception:
            self._failed = True
            logger.exception("Hand runtime failed; disabling hand control")
            try:
                self._device.open_all(force=True, reason="failure")
            except Exception:
                logger.exception("Failed to open hand after hand runtime failure")
                return ()
            return self._open_pose_commands("failure")

    def close(self) -> tuple[HandPoseCommand, ...]:
        try:
            self._mapper.close()
        finally:
            self._device.close()
        return self._open_pose_commands("shutdown")

    def _open_pose_commands(self, reason: str) -> tuple[HandPoseCommand, ...]:
        return tuple(
            HandPoseCommand(command.side, command.pose, True, reason)
            for command in self._open_commands
        )


class DisabledHandRuntime:
    enabled = False

    def start(self) -> tuple[HandPoseCommand, ...]:
        return ()

    def tick(
        self,
        *,
        controller_snapshot: object | None,
        hand_snapshot: object | None,
        active: bool,
        now_s: float | None = None,
    ) -> tuple[HandPoseCommand, ...]:
        del controller_snapshot, hand_snapshot, active, now_s
        return ()

    def close(self) -> tuple[HandPoseCommand, ...]:
        return ()


def build_hand_runtime(cfg: Any) -> HandRuntime | DisabledHandRuntime:
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    if not bool(cfg_get(hands_cfg, "enabled", False)):
        return DisabledHandRuntime()
    driver = str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
    if driver == "linkerhand_l6":
        device, mapper = build_linkerhand_l6(cfg)
    elif driver == "linkerhand_o6":
        device, mapper = build_linkerhand_o6(cfg)
    else:
        raise ValueError(f"Unsupported hands.driver={driver!r}; supported drivers: linkerhand_l6, linkerhand_o6")
    return HandRuntime(device, mapper, open_commands=_open_commands_from_device(device))


def _open_commands_from_device(device: HandDevice) -> tuple[HandPoseCommand, ...]:
    config = getattr(device, "config", None)
    sides = tuple(str(side).strip().lower() for side in getattr(config, "sides", ()))
    open_pose = tuple(int(value) for value in getattr(config, "open_pose", ()))
    if not sides or len(open_pose) != 6:
        return ()
    return tuple(HandPoseCommand(side, open_pose, True, "open") for side in sides)
