---
sidebar_position: 3
---

# Standalone Standing Test

Run this before full sim2real control when bringing up a new robot, network
setup, or policy. It verifies the G1 bridge and the same RL standing path used
by sim2real, without Pico, BVH playback, retargeting, or the full mocap
pipeline.

```text
G1 LowState -> standing observation -> RL policy -> G1 LowCmd targets
```

## When To Use It

Use the standalone test when:

- you are setting up a new wired or onboard G1 deployment
- `run_sim2real.py` has too many moving parts to debug at once
- you need to verify `g1_bridge_sdk`, network interface selection, and policy
  inference before enabling mocap

## Install Runtime Dependencies

```bash
pip install -e '.[sim2real]'
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

## Dry Run

Use `--dry-run` first for timing checks without sending motor commands:

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface enp130s0 \
    --dry-run
```

## Hardware Standing Test

Run the standing controller after confirming the network interface:

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface enp130s0
```

For onboard deployment, the interface is usually `eth0`:

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0
```

Standalone standing reuses the sim2real standing components: `UnitreeG1Robot`,
`Sim2RealSafetyManager`, `RLPolicyController`, `VelCmdObservationBuilder`, and
`Sim2RealReferenceProcessor`. After locking the current joints, policy targets
are sent while Kp ramps from 10% to the configured gains over 2 seconds. To tune
this startup behavior:

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0 \
    --kp-ramp-duration 2.0 \
    --kp-ramp-floor-ratio 0.1
```

## What It Checks

- `g1_bridge_sdk` imports correctly.
- LowState is received from the robot.
- The dual-input ONNX policy can run the standing observation path.
- Low-level position targets can be published through the C++ bridge.
- Observation construction, action scaling, default standing pose, Kp ramp, and
  joint-limit clipping match the sim2real standing runtime.

## Next Steps

After the standing test works:

- use [Pico Sim2Real](pico-sim2real) for realtime VR teleoperation
- use [BVH Sim2Real Playback](bvh-sim2real) for offline motion playback

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No LowState received | Wrong interface or G1 network not connected | Check Ethernet wiring and `--network-interface` |
| Import error for `g1_bridge_sdk` | Bridge not built or sim2real extra not installed | Run the install and bridge setup commands again |
| Policy shape error | Wrong ONNX export | Use the dual-input TemporalCNN policy exported by `save_onnx.py` |
