---
sidebar_position: 3
---

# Quick Start

This guide walks you through running your first sim2sim playback in under 5 minutes.

## Prerequisites

1. [Install Teleopit](installation) (inference profile)
2. [Download assets](download-assets) (`--only gmr ckpt bvh`)

## Run Offline Sim2Sim

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

You should see MuJoCo viewer windows showing the robot tracking the BVH motion.

## Keyboard Controls

When running with `playback.keyboard.enabled=true`:

| Key | Action |
|-----|--------|
| `Space` / `P` | Pause / Resume |
| `R` | Replay from start |
| `Q` | Stop |

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

## Viewer Modes

Control which viewers are displayed:

```bash
# All viewers (mocap + retarget + sim2sim)
python scripts/run/run_sim.py controller.policy_path=track.onnx viewers=all

# No viewer (headless)
python scripts/run/run_sim.py controller.policy_path=track.onnx viewers=none

# Specific viewers
python scripts/run/run_sim.py controller.policy_path=track.onnx 'viewers=[retarget,sim2sim]'
```

## What's Next

- [Offline Sim2Sim Tutorial](../tutorials/offline-sim2sim) - Full guide with recording and rendering
- [Pico Sim2Sim](../tutorials/pico-sim2sim) - Verify Pico tracking in MuJoCo
- [Standalone Standing](../tutorials/standalone-standing) - Check G1 bridge, network, and policy standing
- [Pico Sim2Real](../tutorials/pico-sim2real) - Deploy Pico teleoperation to Unitree G1
- [BVH Sim2Real](../tutorials/bvh-sim2real) - Replay offline BVH motions on Unitree G1
- [Training](../tutorials/training) - Train your own policy
