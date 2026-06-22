from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline
from teleopit.runtime.cli import validate_policy_path


def _print_sim_controls(cfg: DictConfig) -> None:
    provider = str(cfg.input.get("provider", "bvh")).lower()
    if provider == "pico4":
        print("Pico sim2sim controls:")
        if bool(cfg.get("keyboard", {}).get("enabled", False)):
            print("  Keyboard: starts in STANDING; Y mocap, A pause/resume, B arms, X standing, Q quit.")
        else:
            print("  Pico controller: A pause/resume, B arms.")
        print("  State flow: STANDING -> MOCAP <-> ARMS, X -> STANDING.")
        return
    if bool(cfg.get("playback", {}).get("keyboard", {}).get("enabled", False)):
        print("Offline sim2sim controls:")
        print("  Keyboard: Space/P pause/resume, R replay, Q stop.")


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="default")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_sim.py")
    pipeline = TeleopPipeline(cfg)
    num_steps = int(cfg.get("num_steps", 0))
    if cfg.input.get("provider") == "pico4":
        print("Waiting for Pico4 body tracking data...")
    _print_sim_controls(cfg)
    result = pipeline.run(num_steps=num_steps)
    print(result)


if __name__ == "__main__":
    main()
