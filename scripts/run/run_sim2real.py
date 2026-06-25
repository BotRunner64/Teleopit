"""G1 sim2real control entry point — standing/mocap dual mode."""

from __future__ import annotations

import inspect

import hydra
from omegaconf import DictConfig

from teleopit.runtime.common import cfg_get
from teleopit.runtime.console import (
    PlainConsole,
    configure_runtime_logging,
    sim2real_operator_controls,
)
from teleopit.runtime.cli import validate_policy_path
from teleopit.sim2real.mp import Sim2RealRuntime


def _sim2real_status(cfg: DictConfig) -> tuple[tuple[str, str], ...]:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    provider = str(cfg_get(input_cfg, "provider", "bvh")).lower()
    input_label = "Pico4 live" if provider == "pico4" else "BVH"
    recording_cfg = cfg_get(cfg, "recording", {}) or {}
    recording = "enabled" if bool(cfg_get(recording_cfg, "enabled", False)) else "off"
    return (
        ("State", "IDLE"),
        ("Runtime", "multiprocess"),
        ("Input", input_label),
        ("Recording", recording),
    )


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="sim2real")
def main(cfg: DictConfig) -> None:
    _run_sim2real(cfg)


def _run_sim2real(cfg: DictConfig) -> None:
    configure_runtime_logging(cfg, force=True)
    validate_policy_path(cfg, "run_sim2real.py")
    console = PlainConsole(title="Teleopit sim2real")
    runtime_params = inspect.signature(Sim2RealRuntime).parameters
    controller = Sim2RealRuntime(cfg, console=console) if "console" in runtime_params else Sim2RealRuntime(cfg)
    events = []
    input_cfg = cfg_get(cfg, "input", {}) or {}
    if cfg_get(input_cfg, "provider", None) == "pico4":
        events.append("waiting for Pico4 body tracking data")
    console.start(
        status=_sim2real_status(cfg),
        controls=sim2real_operator_controls(cfg),
        events=events,
        control_section="Controls",
        show_help_key=False,
    )
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
