---
sidebar_position: 1
---

# Configuration Overview

Teleopit uses [Hydra](https://hydra.cc/) for composable configuration. Most entry points start from a top-level YAML and allow command-line overrides.

Runtime assembly is centralized in `teleopit/runtime/`. Scripts, `TeleopPipeline`, and the sim2real state machine all share the same path resolution, default propagation, and dimension validation logic.

## Top-Level Configs

| Config | Use Case |
|--------|----------|
| `teleopit/configs/default.yaml` | Offline sim2sim |
| `teleopit/configs/sim2real.yaml` | Unitree G1 hardware control |

These compose sub-configs:

- `teleopit/configs/robot/g1.yaml`
- `teleopit/configs/controller/rl_policy.yaml`
- `teleopit/configs/input/bvh.yaml`

## Override Examples

### Basic Sim2Sim

```bash
python scripts/run/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    policy_hz=50 \
    pd_hz=1000
```

### Change Viewers

```bash
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=none
python scripts/run/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
```

### Enable Keyboard Playback

```bash
python scripts/run/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

### Change Network Interface (sim2real)

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=policy.onnx \
    real_robot.network_interface=enp3s0
```

## Design Principle: Fail-Fast

Teleopit does not silently fix misconfigurations:

- Wrong policy dimensions -> error
- Observation definition mismatch -> error
- Missing required paths -> error
- Using deprecated config key `viewer` -> error
- No auto-padding/trimming of observations

When you encounter a configuration error, look for **which two components have inconsistent definitions**.

For the complete field reference, see [Config Reference](config-reference).
