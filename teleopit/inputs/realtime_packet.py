from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar


FrameT = TypeVar("FrameT")


class ControlEventType(str, Enum):
    TOGGLE_PAUSE = "toggle_pause"


@dataclass(frozen=True)
class ControlEvent:
    event_type: ControlEventType
    source: str
    timestamp_s: float | None = None


@dataclass(frozen=True)
class RealtimeInputPacket(Generic[FrameT]):
    frame: FrameT
    timestamp_s: float
    seq: int
    control_events: tuple[ControlEvent, ...] = ()
