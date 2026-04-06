"""G1 sim2real control entry point — standing/mocap dual mode."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.runtime.cli import validate_policy_path
from teleopit.sim2real.controller import Sim2RealController


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="sim2real")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_sim2real.py")
    controller = Sim2RealController(cfg)
    if cfg.input.get("provider") == "pico4":
        print("Waiting for Pico4 body tracking data...")
    elif cfg.input.get("provider") == "bvh":
        print("Offline playback controls: Y start, A pause/resume, B replay, X standing, L1+R1 estop.")
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
