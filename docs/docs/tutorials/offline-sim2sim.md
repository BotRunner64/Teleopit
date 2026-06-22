---
sidebar_position: 1
---

# Offline Sim2Sim

Run BVH motion capture files through the RL policy in MuJoCo simulation.

## Basic Playback

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

### Using hc_mocap Format

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/hc_mocap/walk.bvh \
    input.bvh_format=hc_mocap
```

## Keyboard Playback

Enable interactive control for offline BVH playback:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    playback.keyboard.enabled=true
```

| Key | Action |
|-----|--------|
| `Space` / `P` | Pause / Resume |
| `R` | Replay from start |
| `Q` | Stop |

Additional options:

```bash
# Pause at end of motion
playback.pause_on_end=true

# Limit number of steps (0 = infinite)
num_steps=300

# Wall-clock rate limiting (even without viewer)
realtime=true
```

## Viewer Modes

Viewers run in separate subprocesses. Use shell quotes for list overrides.

```bash
viewers=sim2sim          # Default
viewers=all              # mocap + retarget + sim2sim
viewers=none             # Headless
'viewers=[retarget,sim2sim]'  # Specific combination
```

:::note
When all active viewer windows are closed, the simulation ends automatically.
:::

## Offline Rendering

Render simulation to video (headless):

```bash
MUJOCO_GL=egl python scripts/render/render_sim.py \
    --bvh data/sample_bvh/aiming1_subject1.bvh \
    --policy track.onnx
```

For hc_mocap format:

```bash
MUJOCO_GL=egl python scripts/render/render_sim.py \
    --bvh data/hc_mocap/wander.bvh \
    --format hc_mocap \
    --policy track.onnx
```

The render pipeline outputs three views (mocap input, retarget, sim2sim), all using MuJoCo rendering.
