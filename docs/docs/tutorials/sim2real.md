---
sidebar_position: 99
---

# Sim2Real Overview

Sim2Real control uses `g1_bridge_sdk`, the C++ DDS bridge for Unitree G1. Use
this page to choose the right hardware tutorial.

| Goal | Tutorial |
|------|----------|
| Verify bridge, network, and policy standing without mocap | [Standalone Standing Test](standalone-standing) |
| Drive the physical G1 from Pico full-body tracking | [Pico Sim2Real](pico-sim2real) |
| Replay an offline BVH motion on the physical G1 | [BVH Sim2Real Playback](bvh-sim2real) |

## Shared Concepts

`real_robot.network_interface` must point to the Linux interface used for
Unitree DDS communication.

| Deployment | Where Teleopit Runs | Typical Interface |
|------------|---------------------|-------------------|
| Wired PC-to-G1 | External PC connected to G1 by Ethernet | `enp...`, for example `enp130s0` |
| Onboard | G1 onboard computer | `eth0` |

The Unitree remote controls the real-robot state machine:

| Button | Action |
|--------|--------|
| `Start` | Enter `STANDING` from `IDLE` or `DAMPING` |
| `Y` | Enter `MOCAP` from `STANDING` |
| `X` | Return from `MOCAP` to `STANDING` |
| `L1+R1` | Emergency stop into `DAMPING` |

Always verify the policy in simulation first, keep the Unitree remote in hand,
and enter `MOCAP` only after input data is stable.
