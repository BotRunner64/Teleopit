---
sidebar_position: 3
---

# Sim2Real Deployment

Deploy Teleopit to control a physical Unitree G1 robot via g1_bridge_sdk (C++ DDS bridge).

:::tip
For Pico 4 / Pico 4 Ultra VR deployment, see the complete [Pico VR Tutorial](pico4-vr).
:::

## Input Sources

| Source | Config | Documentation |
|--------|--------|---------------|
| **Pico 4 / Pico 4 Ultra** | `--config-name pico4_sim2real` | [Pico VR](pico4-vr) |
| Offline BVH files | Default config | This page |

## Prerequisites

**Hardware:**
- Unitree G1 (29 DOF)
- Unitree wireless remote controller
- Network connection between robot and control PC
- Pico 4 headset, or offline BVH motion files

**Software:**
```bash
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
pip install -e '.[sim2real]'
```

For Pico path, also run: `bash scripts/setup/setup_pico4.sh`

## Offline BVH Playback

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0 \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

### Remote Controller Mapping

| Button | Action |
|--------|--------|
| `Start` | Enter `STANDING` |
| `Y` | Enter playback / `MOCAP` |
| `A` | Pause / Resume |
| `B` | Replay from start |
| `X` | Return to `STANDING` |
| `L1+R1` | Emergency stop (`DAMPING`) |

## Control Modes

| Mode | Data Flow | Use Case |
|------|-----------|----------|
| `STANDING` | Default pose -> RL policy -> joints | Startup, recovery, waiting |
| `MOCAP` | Pico/BVH -> retarget -> RL policy -> joints | Teleoperation / playback |
| `DAMPING` | Send damping command | Emergency stop |

### State Machine

```text
                     +-----------------------------+
                     |    L1+R1 E-stop (any state) |
                     v                             |
  [IDLE] --Start--> [STANDING] --Y--> [MOCAP] --X--> [STANDING]
                           ^                       |
                           +----------Y------------+
    ^                                                  |
    +------------------Start---------------------------+
                           [DAMPING]
```

### Pico MOCAP Sub-states

When using `input.provider=pico4`, the MOCAP mode has additional sub-states:

- **ACTIVE**: Normal live mocap tracking
- **PAUSED**: Freeze reference pose; robot maintains balance but stops following
- **RESUMING**: Clear realtime reference buffer, rebuild yaw/pivot alignment, smooth transition back to live mocap

## Common Parameters

```bash
# Adjust control frequency
policy_hz=30

# Specify BVH file
input.bvh_file=data/sample_bvh/aiming1_subject1.bvh

# Pico timeout
input.pico4_timeout=30

# Pause/resume transition
pause_resume_transition_duration=1.5
pause_resume_warmup_steps=3

# Network interface
real_robot.network_interface=enp3s0
```

## Standalone Standing Test

A minimal script for quick hardware and policy verification, independent of the main framework:

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0
```

Supports `--dry-run` for safe timing benchmarks without sending motor commands.
