from __future__ import annotations

import hydra
from omegaconf import DictConfig

from teleopit.pipeline import TeleopPipeline
from teleopit.runtime.cli import validate_policy_path


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="online")
def main(cfg: DictConfig) -> None:
    validate_policy_path(cfg, "run_online_sim.py")
    pipeline = TeleopPipeline(cfg)
    port = cfg.input.get("udp_port", 1118)
    print(f"Listening for UDP BVH data on port {port} ...")
    try:
        result = pipeline.run(num_steps=int(cfg.get("num_steps", 0)))
        print(result)
    finally:
        if hasattr(pipeline, "input_provider") and hasattr(pipeline.input_provider, "close"):
            pipeline.input_provider.close()


if __name__ == "__main__":
    main()
