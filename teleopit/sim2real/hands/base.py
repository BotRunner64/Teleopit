from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


HAND_SIDES = ("left", "right")


@dataclass(frozen=True)
class HandPoseCommand:
    side: str
    pose: tuple[int, ...]
    force: bool = False
    reason: str = ""


class HandDevice(Protocol):
    def connect(self) -> None: ...

    def send_pose(self, side: str, pose: Sequence[int], *, force: bool = False, reason: str = "") -> None: ...

    def open_all(self, *, force: bool = False, reason: str = "") -> None: ...

    def close(self) -> None: ...


class HandInputMapper(Protocol):
    def start(self) -> None: ...

    def map(self, *, controller_snapshot: object | None, hand_snapshot: object | None, active: bool, now_s: float) -> tuple[HandPoseCommand, ...]: ...

    def close(self) -> None: ...
