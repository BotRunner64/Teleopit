"""G1 sim2real control entry point — standing/mocap dual mode."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.runtime.cli import validate_policy_path
from teleopit.sim2real.controller import Sim2RealController


def _print_sim2real_controls(cfg: DictConfig) -> None:
    provider = str(cfg.input.get("provider", "bvh")).lower()
    print("Sim2real controls:")
    print("  Remote Start: enter STANDING.")
    print("  Remote Y: enter MOCAP.")
    print("  Remote X: return to STANDING.")
    print("  Remote L1+R1: DAMPING / estop.")
    if provider == "pico4":
        print("  Mocap pause/resume: Pico/controller A.")
        print("  Dexterous hand: dexterous_hand.mode=off|gripper|vr_hand_pose (default off).")
    else:
        print("  Offline playback: A pause/resume, B replay from start.")
    print("  State flow: IDLE -> STANDING -> MOCAP -> STANDING, Any -> DAMPING.")


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="sim2real")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_sim2real.py")
    controller = Sim2RealController(cfg)
    if cfg.input.get("provider") == "pico4":
        print("Waiting for Pico4 body tracking data...")
    _print_sim2real_controls(cfg)
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
