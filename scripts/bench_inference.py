"""Benchmark ONNX policy inference speed.

Usage:
    python scripts/bench_inference.py controller.policy_path=track.onnx
    python scripts/bench_inference.py controller.policy_path=track.onnx controller.device=cuda
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from teleopit.runtime.cli import add_unitree_sdk_submodule
add_unitree_sdk_submodule(Path(__file__).resolve().parent.parent)

from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.runtime.common import cfg_get


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="onboard_sim2real")
def main(cfg: DictConfig) -> None:
    robot_cfg = cfg_get(cfg, "robot")
    ctrl_cfg = cfg_get(cfg, "controller")
    num_actions = int(cfg_get(robot_cfg, "num_actions", 29))

    # Build policy
    policy = RLPolicyController(ctrl_cfg)
    obs_dim = policy._expected_obs_dim or 166
    print(f"Policy loaded | obs_dim={obs_dim} | multi_input={policy._multi_input}")
    if policy._multi_input:
        print(f"  history_length={policy._history_length} history_obs_dim={policy._history_obs_dim}")

    # Detect provider
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"Available ONNX providers: {providers}")
    device = str(cfg_get(ctrl_cfg, "device", "cpu"))
    print(f"Configured device: {device}")

    # Warmup
    obs = np.random.randn(obs_dim).astype(np.float32) * 0.1
    print("\nWarming up (20 iterations)...")
    for _ in range(20):
        policy.compute_action(obs)

    # Benchmark
    for n_iters in [100, 500]:
        obs_batch = [np.random.randn(obs_dim).astype(np.float32) * 0.1 for _ in range(n_iters)]

        t0 = time.monotonic()
        for i in range(n_iters):
            action = policy.compute_action(obs_batch[i])
            _ = policy.get_target_dof_pos(action)
        elapsed = time.monotonic() - t0

        avg_ms = elapsed / n_iters * 1000
        max_hz = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
        print(f"\n--- {n_iters} iterations ---")
        print(f"  Total: {elapsed:.3f}s")
        print(f"  Average: {avg_ms:.2f} ms/iter")
        print(f"  Max throughput: {max_hz:.1f} Hz")
        print(f"  50Hz budget (20ms): {'OK' if avg_ms < 20 else 'OVER'} ({avg_ms:.2f}ms)")

    # Also benchmark the full observation build + inference pipeline time
    print("\n--- Full step simulation (obs build + inference + post-process) ---")
    from teleopit.controllers.observation import VelCmdObservationBuilder
    from teleopit.interfaces import RobotState

    obs_builder = VelCmdObservationBuilder(cfg)
    default_angles = np.asarray(cfg_get(robot_cfg, "default_angles"), dtype=np.float32)

    # Fake robot state
    fake_state = RobotState(
        qpos=default_angles.copy(),
        qvel=np.zeros(num_actions, dtype=np.float32),
        quat=np.array([1, 0, 0, 0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=time.time(),
    )
    # Fake motion qpos
    motion_qpos = np.zeros(7 + num_actions, dtype=np.float32)
    motion_qpos[3] = 1.0  # identity quat
    motion_qpos[7:] = default_angles
    last_action = np.zeros(num_actions, dtype=np.float32)

    # Warmup
    for _ in range(10):
        o = obs_builder.build(
            fake_state, motion_qpos,
            np.zeros(num_actions, dtype=np.float32),
            last_action,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        a = policy.compute_action(o)
        policy.get_target_dof_pos(a)

    n_iters = 200
    t0 = time.monotonic()
    for _ in range(n_iters):
        o = obs_builder.build(
            fake_state, motion_qpos,
            np.zeros(num_actions, dtype=np.float32),
            last_action,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        a = policy.compute_action(o)
        target = policy.get_target_dof_pos(a)
        last_action = a
    elapsed = time.monotonic() - t0
    avg_ms = elapsed / n_iters * 1000
    max_hz = 1000.0 / avg_ms if avg_ms > 0 else float("inf")

    print(f"  {n_iters} iterations, {elapsed:.3f}s total")
    print(f"  Average: {avg_ms:.2f} ms/step")
    print(f"  Max throughput: {max_hz:.1f} Hz")
    print(f"  50Hz budget (20ms): {'OK' if avg_ms < 20 else 'OVER'} ({avg_ms:.2f}ms)")


if __name__ == "__main__":
    main()
