"""G1 onboard sim2real control entry point.

Receives Pico4 tracking data over ZMQ from the upper machine,
runs retargeting + RL policy + DDS robot control locally on the
G1 onboard computer.

Usage (on G1 onboard computer):
    python scripts/run_onboard_sim2real.py controller.policy_path=policy.onnx
    python scripts/run_onboard_sim2real.py controller.policy_path=policy.onnx input.zmq_host=192.168.1.100
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from teleopit.runtime.cli import add_unitree_sdk_submodule, validate_policy_path

add_unitree_sdk_submodule(Path(__file__).resolve().parent.parent)
from teleopit.sim2real.controller import Sim2RealController


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="onboard_sim2real")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_onboard_sim2real.py")
    controller = Sim2RealController(cfg)
    print("Waiting for ZMQ body tracking data from upper machine...")
    try:
        controller.run()
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
