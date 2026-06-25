from __future__ import annotations

import sys
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from teleopit.runtime.common import cfg_get


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
OPERATOR_LOGGER_NAME = "teleopit.operator"


@dataclass(frozen=True)
class KeyboardControl:
    keys: str
    action: str


def console_show_timing(cfg: Any) -> bool:
    console_cfg = cfg_get(cfg, "console", {}) or {}
    return bool(cfg_get(console_cfg, "show_timing", False))


def console_timing_interval_s(cfg: Any, default: float = 10.0) -> float:
    console_cfg = cfg_get(cfg, "console", {}) or {}
    return float(cfg_get(console_cfg, "timing_log_interval_s", default))


def console_log_level(cfg: Any, default: str = "warning") -> str:
    console_cfg = cfg_get(cfg, "console", {}) or {}
    return str(cfg_get(console_cfg, "log_level", default)).strip().lower()


def configure_runtime_logging(cfg: Any, *, force: bool = False) -> None:
    """Keep operator output quiet by default while preserving warnings/errors."""

    level_name = console_log_level(cfg)
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s", force=force)

    # Operator events are intentional console feedback; ordinary subsystem INFO stays hidden.
    logging.getLogger(OPERATOR_LOGGER_NAME).setLevel(logging.INFO)

    if level > logging.INFO:
        noisy_names = (
            "pico_bridge",
            "onnxruntime",
            "teleopit.inputs.pico4_provider",
            "teleopit.sim2real.unitree_g1",
            "teleopit.sim2real.safety",
        )
        for name in noisy_names:
            logging.getLogger(name).setLevel(logging.WARNING)


class PlainConsole:
    """Small runtime console for keyboard controls and operator feedback."""

    def __init__(self, *, title: str, enabled: bool = True, color: bool | None = None) -> None:
        self._title = str(title)
        self._enabled = bool(enabled)
        self._color = sys.stdout.isatty() if color is None else bool(color)
        self._started = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(
        self,
        *,
        status: Iterable[tuple[str, str]] = (),
        controls: Iterable[KeyboardControl] = (),
        events: Iterable[str] = (),
        control_section: str = "Keyboard",
        show_help_key: bool = True,
    ) -> None:
        if not self._enabled:
            return
        self._started = True
        print(
            self.render(
                status=status,
                controls=controls,
                events=events,
                control_section=control_section,
                show_help_key=show_help_key,
            ),
            flush=True,
        )

    def event(self, message: str) -> None:
        if not self._enabled:
            return
        prefix = datetime.now().strftime("%H:%M:%S")
        print(f"{self._dim(prefix)} {self._highlight_text(message)}", flush=True)

    def key_feedback(self, key: str, action: str, *, result: str | None = None) -> None:
        details = f" -> {result}" if result else ""
        self.event(f"{self.format_key(key)} {action}{details}")

    def help(self, controls: Iterable[KeyboardControl]) -> None:
        if not self._enabled:
            return
        control_items = list(controls)
        if not control_items:
            self.event("no keyboard controls active")
            return
        lines = [self._section("Keyboard help")]
        lines.extend(f"  {self.format_key(item.keys)} {item.action}" for item in control_items)
        print("\n".join(lines), flush=True)

    def render(
        self,
        *,
        status: Iterable[tuple[str, str]] = (),
        controls: Iterable[KeyboardControl] = (),
        events: Iterable[str] = (),
        control_section: str = "Keyboard",
        show_help_key: bool = True,
    ) -> str:
        lines = [self._title_text(self._title)]
        status_items = [(str(key), str(value)) for key, value in status if str(value)]
        if status_items:
            width = max(len(key) for key, _value in status_items)
            lines.append("")
            lines.extend(
                f"{self._label(key.ljust(width))}  {self._status_value(value)}"
                for key, value in status_items
            )

        control_items = list(controls)
        if control_items:
            lines.append("")
            lines.append(self._section(control_section))
            lines.append("   ".join(f"{self.format_key(item.keys)} {item.action}" for item in control_items))
            if show_help_key:
                lines.append(f"{self.format_key('H')} help")

        event_items = [str(event) for event in events if str(event)]
        if event_items:
            lines.append("")
            lines.append(self._section("Events"))
            lines.extend(self._highlight_text(event) for event in event_items)
        return "\n".join(lines)

    def _style(self, text: str, code: str) -> str:
        if not self._color:
            return text
        return f"{code}{text}{RESET}"

    def _title_text(self, text: str) -> str:
        return self._style(text, BOLD)

    def _section(self, text: str) -> str:
        return self._style(text, CYAN + BOLD)

    def _label(self, text: str) -> str:
        return self._style(text, DIM)

    def _dim(self, text: str) -> str:
        return self._style(text, DIM)

    def format_key(self, text: str) -> str:
        return self._style(f"[{text}]", YELLOW + BOLD)

    def _status_value(self, text: str) -> str:
        normalized = text.strip().lower()
        if normalized in {"ok", "enabled", "running", "mocap", "standing", "active"}:
            return self._style(text, GREEN + BOLD)
        if normalized in {"idle", "waiting", "paused", "off", "none"} or "waiting" in normalized:
            return self._style(text, YELLOW + BOLD)
        if normalized in {"damping", "error", "failed"} or "stale" in normalized:
            return self._style(text, RED + BOLD)
        if normalized in {"arms", "pico4 live"}:
            return self._style(text, MAGENTA + BOLD)
        return self._style(text, BOLD)

    def _highlight_text(self, text: str) -> str:
        if not self._color:
            return text
        highlighted = text
        replacements = {
            "waiting": YELLOW + BOLD,
            "paused": YELLOW + BOLD,
            "resumed": GREEN + BOLD,
            "replay": GREEN + BOLD,
            "stopping": RED + BOLD,
            "shutdown": RED + BOLD,
            "MOCAP": GREEN + BOLD,
            "STANDING": GREEN + BOLD,
            "ARMS": MAGENTA + BOLD,
        }
        for word, code in replacements.items():
            highlighted = highlighted.replace(word, f"{code}{word}{RESET}")
        return highlighted


def sim_keyboard_controls(cfg: Any) -> tuple[KeyboardControl, ...]:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    provider = str(cfg_get(input_cfg, "provider", "bvh")).lower()
    if provider == "pico4":
        keyboard_cfg = cfg_get(cfg, "keyboard", {}) or {}
        if not bool(cfg_get(keyboard_cfg, "enabled", False)):
            return ()
        return (
            KeyboardControl("Y", "mocap"),
            KeyboardControl("A", "pause/resume"),
            KeyboardControl("B", "arms"),
            KeyboardControl("X", "standing"),
            KeyboardControl("Q", "quit"),
        )

    playback_cfg = cfg_get(cfg, "playback", {}) or {}
    keyboard_cfg = cfg_get(playback_cfg, "keyboard", {}) or {}
    if not bool(cfg_get(keyboard_cfg, "enabled", False)):
        return ()
    return (
        KeyboardControl("Space/P", "pause/resume"),
        KeyboardControl("R", "replay"),
        KeyboardControl("Q", "stop"),
    )


def sim2real_keyboard_controls(cfg: Any) -> tuple[KeyboardControl, ...]:
    recording_cfg = cfg_get(cfg, "recording", {}) or {}
    if not bool(cfg_get(recording_cfg, "enabled", False)):
        return ()
    return (
        KeyboardControl("R", "start"),
        KeyboardControl("S", "save"),
        KeyboardControl("D", "discard"),
        KeyboardControl("Q", "shutdown"),
    )


def sim2real_operator_controls(cfg: Any) -> tuple[KeyboardControl, ...]:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    provider = str(cfg_get(input_cfg, "provider", "bvh")).lower()
    controls = [
        KeyboardControl("Remote Start", "standing"),
        KeyboardControl("Remote Y", "mocap"),
        KeyboardControl("Remote X", "standing"),
        KeyboardControl("Remote L1+R1", "damping / estop"),
    ]
    if provider == "pico4":
        controls.extend(
            [
                KeyboardControl("Pico/Controller A", "pause/resume"),
                KeyboardControl("Pico/Controller B", "arms"),
            ]
        )
    else:
        controls.extend(
            [
                KeyboardControl("Remote A", "pause/resume playback"),
                KeyboardControl("Remote B", "replay from start"),
            ]
        )
    controls.extend(sim2real_keyboard_controls(cfg))
    return tuple(controls)
