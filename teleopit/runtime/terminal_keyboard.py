from __future__ import annotations

import os
import select
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalKeyEvent:
    key: str


class TerminalKeyboardReader:
    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs: list[object] | None = None
        if not sys.stdin.isatty():
            return
        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self.close()

    @property
    def active(self) -> bool:
        return self._fd is not None and self._old_attrs is not None

    def poll(self) -> tuple[TerminalKeyEvent, ...]:
        if not self.active or self._fd is None:
            return ()
        ready, _, _ = select.select([self._fd], [], [], 0.0)
        if not ready:
            return ()

        events: list[TerminalKeyEvent] = []
        while True:
            try:
                chars = os.read(self._fd, 32)
            except BlockingIOError:
                break
            if not chars:
                break
            for char in chars.decode("utf-8", errors="ignore"):
                if char:
                    events.append(TerminalKeyEvent(key=char))
            ready, _, _ = select.select([self._fd], [], [], 0.0)
            if not ready:
                break
        return tuple(events)

    def close(self) -> None:
        if self._fd is None or self._old_attrs is None:
            self._fd = None
            self._old_attrs = None
            return
        try:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        finally:
            self._fd = None
            self._old_attrs = None
