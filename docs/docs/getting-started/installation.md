---
sidebar_position: 1
---

# Installation

Teleopit supports multiple installation profiles depending on your use case.

## Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/) (recommended)

```bash
conda create -n teleopit python=3.10
conda activate teleopit
```

## Install Profiles

### Inference Only (sim2sim)

```bash
pip install -e .
```

This is sufficient for offline BVH playback and MuJoCo simulation.

### Training

```bash
pip install -e '.[train]'
```

Adds `rsl-rl-lib`, `mjlab`, `wandb`, and training dependencies.

### Sim2Real (Hardware Deployment)

```bash
pip install -e '.[sim2real]'
```

Adds `opencv-python` and `g1_bridge_sdk`. You also need to build the C++ bridge:

```bash
pip install pybind11
pip install third_party/g1_bridge_sdk
```

See [G1 Bridge SDK](../reference/g1-bridge-sdk) for details.

### Pico 4 VR

```bash
pip install -e '.[pico4]'
bash scripts/setup/setup_pico4.sh
```

See [Pico 4 VR Tutorial](../tutorials/pico4-vr) for the full setup guide.

### Onboard (G1 NX)

```bash
bash scripts/setup/setup_onboard.sh
```

This script installs system dependencies, builds `g1_bridge_sdk`, and installs `teleopit[onboard]`.

## Verify Installation

```bash
python -c "import teleopit; print('teleopit OK')"
python -c "import train_mimic.tasks; print('training OK')"  # if training installed
```

## Next Steps

- [Download Assets](download-assets) - Download models and data
- [Quick Start](quick-start) - Run your first simulation
