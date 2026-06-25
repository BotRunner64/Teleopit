from __future__ import annotations

import logging

from teleopit.runtime.console import (
    OPERATOR_LOGGER_NAME,
    PlainConsole,
    configure_runtime_logging,
    sim2real_operator_controls,
    sim_keyboard_controls,
)


def test_sim_console_shows_only_enabled_keyboard_controls() -> None:
    cfg = {
        "input": {"provider": "pico4"},
        "keyboard": {"enabled": True},
    }

    labels = [control.keys for control in sim_keyboard_controls(cfg)]

    assert labels == ["Y", "A", "B", "X", "Q"]


def test_sim_console_hides_non_keyboard_controls() -> None:
    cfg = {
        "input": {"provider": "pico4"},
        "keyboard": {"enabled": False},
    }

    assert sim_keyboard_controls(cfg) == ()


def test_sim2real_console_shows_remote_and_pico_controls() -> None:
    cfg = {"input": {"provider": "pico4"}, "recording": {"enabled": False}}

    rendered = PlainConsole(title="Teleopit sim2real", color=False).render(
        controls=sim2real_operator_controls(cfg),
        control_section="Controls",
        show_help_key=False,
    )

    assert "Controls" in rendered
    assert "[Remote Start] standing" in rendered
    assert "[Remote L1+R1] damping / estop" in rendered
    assert "[Pico/Controller A] pause/resume" in rendered
    assert "[Pico/Controller B] arms" in rendered
    assert "[H] help" not in rendered


def test_sim2real_console_includes_terminal_recording_controls_when_enabled() -> None:
    cfg = {"recording": {"enabled": True}}

    rendered = PlainConsole(title="Teleopit sim2real", color=False).render(
        controls=sim2real_operator_controls(cfg),
        control_section="Controls",
        show_help_key=False,
    )

    assert "Controls" in rendered
    assert "[Remote Start] standing" in rendered
    assert "[R] start" in rendered
    assert "[Q] shutdown" in rendered
    assert "[H] help" not in rendered


def test_console_can_highlight_important_words_with_ansi_color() -> None:
    rendered = PlainConsole(title="Teleopit sim2sim", color=True).render(
        status=(("State", "MOCAP"),),
        controls=sim_keyboard_controls({"input": {"provider": "bvh"}, "playback": {"keyboard": {"enabled": True}}}),
        events=("waiting for input",),
    )

    assert "\033[" in rendered
    assert "[Space/P]" in rendered
    assert "MOCAP" in rendered


def test_runtime_logging_keeps_operator_info_but_hides_noisy_info() -> None:
    configure_runtime_logging({"console": {"log_level": "warning"}}, force=True)

    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
    assert logging.getLogger(OPERATOR_LOGGER_NAME).getEffectiveLevel() == logging.INFO
    assert logging.getLogger("pico_bridge").getEffectiveLevel() == logging.WARNING
