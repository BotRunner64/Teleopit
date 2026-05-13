---
sidebar_position: 5
---

# BVH Playback on Unitree G1

Use this tutorial to replay an offline BVH motion on a physical Unitree G1. This
path does not use Pico tracking.

```text
BVH file -> retarget -> RL policy -> g1_bridge_sdk -> G1
```

For realtime Pico control, use [Pico Sim2Real](pico-sim2real).

## 1. Install Runtime Dependencies

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

## 2. Choose The Network Interface

For wired PC-to-G1 deployment, connect the PC to the robot by Ethernet, run
`ifconfig`, and use the interface connected to G1:

```bash
real_robot.network_interface=enp130s0
```

For onboard deployment, `eth0` is usually correct.

## 3. Run BVH Playback

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=enp130s0 \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

The default `sim2real.yaml` uses `input.provider=bvh` and
`playback.pause_on_end=true`, so the final pose is held when the motion reaches
the end.

## Remote Controls

| Button | Action |
|--------|--------|
| `Start` | Enter `STANDING` |
| `Y` | Enter playback / `MOCAP` |
| `A` | Pause / resume playback |
| `B` | Replay from frame 0 |
| `X` | Return to `STANDING` |
| `L1+R1` | Emergency stop (`DAMPING`) |

## Playback Behavior

- `Y` starts the BVH motion from the current playback position.
- `A` pauses at the current reference pose and resumes from the same timeline.
- `B` resets playback to frame 0 and restarts the policy/reference state.
- If `playback.pause_on_end=true`, the final pose is held until `B` replays or
  `X` returns to `STANDING`.

## Common Parameters

```bash
# BVH file
input.bvh_file=data/sample_bvh/aiming1_subject1.bvh

# Real G1 DDS interface
real_robot.network_interface=enp130s0

# Pause at the final BVH frame
playback.pause_on_end=true

# Smooth transition from standing/current robot state into playback
transition_duration=2.0

# Resume blend for offline playback
pause_resume_transition_duration=1.0

# Control loop rate
policy_hz=50
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No LowState received | Wrong network interface | Check Ethernet wiring and `real_robot.network_interface` |
| Robot enters `STANDING` but not playback | BVH validation failed | Check `input.bvh_file` and retarget logs |
| Playback ends and robot holds pose | `playback.pause_on_end=true` | Press `B` to replay or `X` for `STANDING` |
| `B` does nothing | Not in offline BVH MOCAP mode | Enter `MOCAP` with `Y` first |
