<p align="center">
  <img src="assets/teleopit_logo.jpg" width="80" alt="Teleopit">
</p>

<p align="center">
  <h1 align="center">Teleopit</h1>
  <h3 align="center">Lightweight, extensible whole-body teleoperation framework for humanoid robots</h3>
</p>

<p align="center">
  <a href="https://BotRunner64.github.io/Teleopit/">Documentation</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/docs/getting-started/quick-start">Quick Start</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/docs/tutorials/pico4-vr">Pico VR</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/docs/tutorials/sim2real">Sim2Real</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/docs/tutorials/training">Training</a>
</p>

## Installation

```bash
pip install -e .              # Inference (sim2sim)
pip install -e '.[train]'     # Training
pip install -e '.[sim2real]'  # Hardware deployment
```

## Download Models and Data

```bash
pip install modelscope
python scripts/setup/download_assets.py
```

Download only inference essentials:

```bash
python scripts/setup/download_assets.py --only gmr ckpt bvh
```

## Quick Start

Offline sim2sim:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

Training:

```bash
python train_mimic/scripts/train.py \
    --motion_file data/datasets/seed/train \
    --num_envs 4096 \
    --max_iterations 30000
```

Export ONNX:

```bash
python train_mimic/scripts/save_onnx.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --output track.onnx \
    --history_length 10
```

## Use Cases

| Scenario | Command | Docs |
|----------|---------|------|
| **Offline sim2sim** | `python scripts/run/run_sim.py controller.policy_path=track.onnx input.bvh_file=...` | [Sim2Sim](https://BotRunner64.github.io/Teleopit/docs/tutorials/offline-sim2sim) |
| **Pico 4 VR teleoperation** | `python scripts/run/run_sim.py --config-name pico4_sim ...` | [Pico VR](https://BotRunner64.github.io/Teleopit/docs/tutorials/pico4-vr) |
| **Pico 4 hardware deploy** | `python scripts/run/run_sim2real.py --config-name pico4_sim2real ...` | [Pico VR](https://BotRunner64.github.io/Teleopit/docs/tutorials/pico4-vr) |
| **G1 Onboard (NX)** | `python scripts/run/run_onboard_sim2real.py ...` | [Onboard](https://BotRunner64.github.io/Teleopit/docs/tutorials/onboard-sim2real) |
| **Offline keyboard replay** | `python scripts/run/run_sim.py ... playback.keyboard.enabled=true` | [Sim2Sim](https://BotRunner64.github.io/Teleopit/docs/tutorials/offline-sim2sim) |
| **Training & export** | `python train_mimic/scripts/train.py ...` | [Training](https://BotRunner64.github.io/Teleopit/docs/tutorials/training) |

Offline sim2sim keyboard: `Space/P` pause/resume, `R` replay, `Q` stop. Sim2real uses the wireless remote: `Y` enter playback, `A` pause/resume, `B` replay, `X` return to standing.

## Documentation

Full documentation is available at **[BotRunner64.github.io/Teleopit](https://BotRunner64.github.io/Teleopit/)**.

- [Installation](https://BotRunner64.github.io/Teleopit/docs/getting-started/installation)
- [Configuration](https://BotRunner64.github.io/Teleopit/docs/configuration/overview)
- [Architecture](https://BotRunner64.github.io/Teleopit/docs/reference/architecture)
- [Asset Management](https://BotRunner64.github.io/Teleopit/docs/reference/assets)

## Changelog

### v0.2.0 (2026-04-03)

Onboard Sim2Real (G1 NX + ZMQ Pico4), G1 Bridge SDK (C++ DDS pybind11), standalone Standing controller (RL policy + timing diagnostics), realtime mocap buffering with catch-up optimization; released model upgraded to 30k checkpoint.

### v0.1.1 (2025-03-28)

Dataset shard-only refactor, adaptive_bin sampling, external asset management, repository slimming.

### v0.1.0 (2025-03-25)

Initial public release: General-Tracking-G1 whole-body tracking training, ONNX sim2sim inference, Pico 4 VR teleoperation, Unitree G1 hardware deployment.

## License

[MIT](LICENSE)
