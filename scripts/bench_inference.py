"""Benchmark ONNX policy inference and full control step speed.

Usage:
    python scripts/bench_inference.py controller.policy_path=track.onnx
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


def _bench(label: str, func, n_iters: int) -> float:
    """Run func n_iters times, print stats, return avg_ms."""
    t0 = time.monotonic()
    for _ in range(n_iters):
        func()
    elapsed = time.monotonic() - t0
    avg_ms = elapsed / n_iters * 1000
    max_hz = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    budget_ok = "OK" if avg_ms < 20 else "OVER"
    print(f"  [{label}] {n_iters} iters | avg {avg_ms:.2f} ms | {max_hz:.1f} Hz | 50Hz budget: {budget_ok}")
    return avg_ms


@hydra.main(version_base=None, config_path="../teleopit/configs", config_name="onboard_sim2real")
def main(cfg: DictConfig) -> None:
    robot_cfg = cfg_get(cfg, "robot")
    ctrl_cfg = cfg_get(cfg, "controller")
    num_actions = int(cfg_get(robot_cfg, "num_actions", 29))

    # --- 1. Policy inference benchmark ---
    print("=" * 60)
    print("1. ONNX Policy Inference")
    print("=" * 60)

    policy = RLPolicyController(ctrl_cfg)
    obs_dim = policy._expected_obs_dim or 166
    print(f"  obs_dim={obs_dim} | multi_input={policy._multi_input}")
    if policy._multi_input:
        print(f"  history_length={policy._history_length} history_obs_dim={policy._history_obs_dim}")

    import onnxruntime as ort
    print(f"  ONNX providers: {ort.get_available_providers()}")
    print(f"  device: {cfg_get(ctrl_cfg, 'device', 'cpu')}")

    # Warmup
    obs = np.random.randn(obs_dim).astype(np.float32) * 0.1
    for _ in range(20):
        policy.compute_action(obs)

    obs_list = [np.random.randn(obs_dim).astype(np.float32) * 0.1 for _ in range(500)]
    idx = [0]

    def infer_only():
        a = policy.compute_action(obs_list[idx[0] % 500])
        policy.get_target_dof_pos(a)
        idx[0] += 1

    _bench("inference", infer_only, 100)
    idx[0] = 0
    _bench("inference", infer_only, 500)

    # --- 2. Full step (obs build + inference + post-process) ---
    print()
    print("=" * 60)
    print("2. Full Control Step (obs_build + inference + post_process)")
    print("=" * 60)

    from teleopit.controllers.observation import VelCmdObservationBuilder
    from teleopit.interfaces import RobotState

    obs_cfg = {
        "num_actions": int(cfg_get(robot_cfg, "num_actions")),
        "default_dof_pos": list(cfg_get(robot_cfg, "default_angles")),
        "xml_path": str(cfg_get(robot_cfg, "xml_path")),
        "anchor_body_name": cfg_get(robot_cfg, "anchor_body_name", "torso_link"),
    }
    obs_builder = VelCmdObservationBuilder(obs_cfg)

    default_angles = np.asarray(cfg_get(robot_cfg, "default_angles"), dtype=np.float32)
    fake_state = RobotState(
        qpos=default_angles.copy(),
        qvel=np.zeros(num_actions, dtype=np.float32),
        quat=np.array([1, 0, 0, 0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=time.time(),
    )
    motion_qpos = np.zeros(7 + num_actions, dtype=np.float32)
    motion_qpos[3] = 1.0
    motion_qpos[7:] = default_angles
    last_action = np.zeros(num_actions, dtype=np.float32)

    # Warmup
    for _ in range(10):
        o = obs_builder.build(
            fake_state, motion_qpos,
            np.zeros(num_actions, dtype=np.float32), last_action,
            np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        )
        a = policy.compute_action(o)
        policy.get_target_dof_pos(a)

    def full_step():
        nonlocal last_action
        o = obs_builder.build(
            fake_state, motion_qpos,
            np.zeros(num_actions, dtype=np.float32), last_action,
            np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        )
        a = policy.compute_action(o)
        policy.get_target_dof_pos(a)
        last_action = a

    _bench("full_step", full_step, 200)

    # --- 3. Obs build only ---
    print()
    print("=" * 60)
    print("3. Observation Build Only")
    print("=" * 60)

    def obs_only():
        obs_builder.build(
            fake_state, motion_qpos,
            np.zeros(num_actions, dtype=np.float32), last_action,
            np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        )

    _bench("obs_build", obs_only, 500)

    # --- Summary ---
    print()
    print("=" * 60)
    print("Summary: if all show 'OK', control loop timing is not the issue.")
    print("If onboard still can't recover from pushes, the problem is")
    print("likely DDS state latency or observation content differences.")
    print("=" * 60)


if __name__ == "__main__":
    main()
