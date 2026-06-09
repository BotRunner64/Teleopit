from __future__ import annotations

import logging
import time
from typing import Any

from teleopit.runtime.common import cfg_get
from teleopit.sim2real.hands.base import HandDevice, HandInputMapper
from teleopit.sim2real.hands.linkerhand_l6 import build_linkerhand_l6

logger = logging.getLogger(__name__)


class HandRuntime:
    def __init__(self, device: HandDevice, mapper: HandInputMapper):
        self._device = device
        self._mapper = mapper
        self.enabled = True
        self._failed = False

    def start(self) -> None:
        try:
            self._device.connect()
            self._mapper.start()
        except Exception:
            try:
                self._device.close()
            finally:
                raise

    def tick(self, *, controller_snapshot: object | None, hand_snapshot: object | None, active: bool, now_s: float | None = None) -> None:
        if self._failed:
            return
        now = time.monotonic() if now_s is None else float(now_s)
        try:
            for command in self._mapper.map(
                controller_snapshot=controller_snapshot,
                hand_snapshot=hand_snapshot,
                active=active,
                now_s=now,
            ):
                self._device.send_pose(command.side, command.pose, force=command.force, reason=command.reason)
        except Exception:
            self._failed = True
            logger.exception("Hand runtime failed; disabling hand control")
            try:
                self._device.open_all(force=True, reason="failure")
            except Exception:
                logger.exception("Failed to open hand after hand runtime failure")

    def close(self) -> None:
        try:
            self._mapper.close()
        finally:
            self._device.close()


class DisabledHandRuntime:
    enabled = False

    def start(self) -> None:
        pass

    def tick(self, *, controller_snapshot: object | None, hand_snapshot: object | None, active: bool, now_s: float | None = None) -> None:
        del controller_snapshot, hand_snapshot, active, now_s

    def close(self) -> None:
        pass


def build_hand_runtime(cfg: Any) -> HandRuntime | DisabledHandRuntime:
    hands_cfg = cfg_get(cfg, "hands", {}) or {}
    if not bool(cfg_get(hands_cfg, "enabled", False)):
        return DisabledHandRuntime()
    driver = str(cfg_get(hands_cfg, "driver", "linkerhand_l6")).strip().lower()
    if driver != "linkerhand_l6":
        raise ValueError(f"Unsupported hands.driver={driver!r}; only linkerhand_l6 is implemented")
    device, mapper = build_linkerhand_l6(cfg)
    return HandRuntime(device, mapper)
