---
sidebar_position: 1
slug: /
---

# Introduction

> **Looking for Chinese docs?** [中文文档点此进入](https://BotRunner64.github.io/Teleopit/zh-Hans/)

**Teleopit** is a lightweight, extensible whole-body teleoperation framework for humanoid robots. It provides real-time motion retargeting from human operators to Unitree G1 robots, supporting both MuJoCo simulation and real hardware deployment.

## Key Features

- **Offline sim2sim**: Play back BVH motion capture files through RL policy in MuJoCo
- **VR teleoperation**: Real-time whole-body control via Pico 4 / Pico 4 Ultra full body tracking
- **Sim2real deployment**: Deploy to Unitree G1 hardware with the same pipeline
- **Training pipeline**: End-to-end RL training with General-Tracking-G1 task
- **Extensible design**: Protocol-based components (InputProvider, Retargeter, Controller, Robot)

## Pipeline Overview

```text
InputProvider (BVH / Pico4 VR)
    -> Retargeter (GMR)
    -> ObservationBuilder (167D)
    -> Controller (dual-input TemporalCNN ONNX)
    -> Robot (MuJoCo sim or Unitree G1)
```

## Technical Specs

| Spec | Value |
|------|-------|
| Policy frequency | 50 Hz |
| PD control frequency | 1000 Hz |
| Observation dimension | 167D |
| Action dimension | 29D (G1 joints) |
| ONNX model | Dual-input TemporalCNN |
| Retargeting | GMR (General Motion Retargeting) |
| Simulator | MuJoCo |
| Hardware | Unitree G1 (29 DOF) |

## What's Next

- [Installation](getting-started/installation) - Set up your environment
- [Quick Start](getting-started/quick-start) - Run your first sim2sim
- [Tutorials](tutorials/offline-sim2sim) - Step-by-step guides for each use case
