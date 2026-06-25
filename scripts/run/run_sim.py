from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline
from teleopit.runtime.common import cfg_get
from teleopit.runtime.console import (
    PlainConsole,
    configure_runtime_logging,
    sim_keyboard_controls,
)
from teleopit.runtime.cli import validate_policy_path


def _sim_status(cfg: DictConfig) -> tuple[tuple[str, str], ...]:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    provider = str(cfg_get(input_cfg, "provider", "bvh")).lower()
    viewers = str(cfg_get(cfg, "viewers", "none"))
    if provider == "pico4":
        keyboard_cfg = cfg_get(cfg, "keyboard", {}) or {}
        state = "STANDING" if bool(cfg_get(keyboard_cfg, "enabled", False)) else "MOCAP"
        return (
            ("State", state),
            ("Input", "Pico4 live"),
            ("Viewers", viewers),
        )
    return (
        ("State", "MOCAP"),
        ("Input", "BVH"),
        ("Viewers", viewers),
    )


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="default")
def main(cfg: DictConfig) -> None:
    configure_runtime_logging(cfg, force=True)
    validate_policy_path(cfg, "run_sim.py")
    console = PlainConsole(title="Teleopit sim2sim")
    pipeline = TeleopPipeline(cfg, console=console)
    num_steps = int(cfg.get("num_steps", 0))
    events = []
    input_cfg = cfg_get(cfg, "input", {}) or {}
    if cfg_get(input_cfg, "provider", None) == "pico4":
        events.append("waiting for Pico4 body tracking data")
    console.start(status=_sim_status(cfg), controls=sim_keyboard_controls(cfg), events=events)
    result = pipeline.run(num_steps=num_steps)
    console.event(str(result))


if __name__ == "__main__":
    main()
