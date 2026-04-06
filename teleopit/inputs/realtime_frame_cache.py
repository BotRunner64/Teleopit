from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar


FrameT = TypeVar("FrameT")


class RealtimeFrameCache(Generic[FrameT]):
    def __init__(self, *, buffer_size: int, fps_window: int = 30) -> None:
        self._buffer: deque[tuple[FrameT, float]] = deque(maxlen=max(int(buffer_size), 2))
        self._fps_timestamps: deque[float] = deque(maxlen=max(int(fps_window), 2))
        self._frame_seq = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def frame_seq(self) -> int:
        return int(self._frame_seq)

    def fps(self) -> float:
        if len(self._fps_timestamps) < 2:
            return 30.0
        oldest = float(self._fps_timestamps[0])
        newest = float(self._fps_timestamps[-1])
        dt = newest - oldest
        if dt <= 0.0:
            return 30.0
        return (len(self._fps_timestamps) - 1) / dt

    def clear(self) -> None:
        self._buffer.clear()

    def latest(self) -> FrameT:
        if not self._buffer:
            raise RuntimeError("Realtime frame cache is empty")
        return self._buffer[-1][0]

    def latest_packet(self) -> tuple[FrameT, float, int]:
        if not self._buffer:
            raise RuntimeError("Realtime frame cache is empty")
        frame, timestamp = self._buffer[-1]
        return frame, float(timestamp), int(self._frame_seq)

    def snapshot(self) -> list[tuple[FrameT, float]]:
        return list(self._buffer)

    def append(
        self,
        frame: FrameT,
        timestamp: float,
        *,
        fps_timestamp: float | None = None,
        source_seq: int | None = None,
    ) -> int:
        ts = float(timestamp)
        self._buffer.append((frame, ts))
        self._fps_timestamps.append(ts if fps_timestamp is None else float(fps_timestamp))
        if source_seq is None:
            self._frame_seq += 1
        else:
            self._frame_seq = int(source_seq)
        return int(self._frame_seq)
