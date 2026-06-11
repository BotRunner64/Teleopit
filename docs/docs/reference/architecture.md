---
sidebar_position: 1
---

# Architecture

System internals and technical constraints for developers.

## Pipeline

```text
InputProvider (BVH file / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (167D)
    -> Controller (dual-input TemporalCNN ONNX)
    -> Robot (MuJoCo sim or Unitree G1)
```

Offline/online inference is assembled by `teleopit/runtime/` and `teleopit/pipeline.py`. The hardware state machine runs through the process-isolated runtime in `teleopit/sim2real/mp/`. Training is provided by `train_mimic/`.

## Code Structure

```text
configs / scripts
    -> runtime
    -> interfaces + pipeline state machines
    -> adapters (inputs / retargeting / controller / robot / recording)

train_mimic/scripts
    -> train_mimic/app.py
    -> single task registry / env builder / runner cfg
    -> mjlab / rsl_rl

train_mimic/scripts/data
    -> train_mimic/data/dataset_builder.py
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

## Core Boundaries

| Module | Role |
|--------|------|
| `teleopit/interfaces.py` | Stable protocols: InputProvider, Retargeter, Controller, Robot, ObservationBuilder, Recorder |
| `teleopit/runtime/` | Config parsing, path normalization, component assembly, CLI validation |
| `teleopit/pipeline.py` | Lightweight facade for offline sim |
| `teleopit/sim2real/mp/` | Process-isolated sim2real state machine, IPC, and robot-control loop |
| `teleopit/controllers/observation.py` | ObservationBuilder |
| `teleopit/controllers/rl_policy.py` | Accepts dual-input ONNX whose observation dimension matches the runtime builder |
| `train_mimic/app.py` | Shared train/play/benchmark assembly |
| `train_mimic/tasks/tracking/config/` | Single task registration (`General-Tracking-G1`) |
| `train_mimic/data/dataset_builder.py` | Sole official dataset construction entry |

## Technical Specifications

| Spec | Value |
|------|-------|
| Training task | `General-Tracking-G1` |
| Inference observation | `velcmd_history` (167D) |
| ONNX signature | Dual-input `obs` (167D) + `obs_history` |
| Actor/Critic | TemporalCNN (2048, 1024, 512, 256, 128) |
| Training sampling | Default `uniform`; also supports `adaptive` and `rewind`; playback/benchmark use `start` |
| Training `window_steps` | `[0]` |
| Data format | Shard directories only (`shard_*.npz`) |

## Constraints

- `controller.policy_path` must be explicitly provided and the file must exist
- Offline BVH runs require explicit `input.bvh_file`
- `viewers` is the sole viewer configuration entry
- Observation/ONNX dimension mismatch causes immediate startup error
- sim2real also requires a dual-input ONNX whose observation dimension matches the runtime builder

## Public Surface

**Stable run modes:** offline sim2sim, offline sim2real playback, Pico4 sim2sim, G1 sim2real

**Stable training entry points:** `train.py`, `play.py`, `benchmark.py`, `save_onnx.py`

**Stable data entry points:** `build_dataset.py`, `split_shards.py`
