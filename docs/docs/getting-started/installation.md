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

Adds `rsl-rl-lib`, `mjlab`, `wandb`, `swanlab`, and training dependencies.

### Sim2Real (Hardware Deployment)

```bash
pip install -e '.[sim2real]'
```

Adds `opencv-python`. You also need to initialize submodules and build/install the C++ `g1_bridge_sdk` bridge:

```bash
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

See [G1 Bridge SDK](../reference/g1-bridge-sdk) for details.

### Pico 4 VR

```bash
pip install -e '.[pico4]'
```

Teleopit uses the in-process `pico_bridge.PicoBridge` receiver for Pico tracking.
Teleopit targets pico-bridge 0.2.1 and its `pico_native` tracking semantics.
The receiver can run on a workstation PC or the robot onboard computer.
See [Pico Sim2Sim](../tutorials/pico-sim2sim) and
[Pico Sim2Real](../tutorials/pico-sim2real) for the full setup guides.

Optional LinkerHand control for Pico sim2real uses local third-party packages.
Install those packages directly after initializing the submodules:

```bash
git submodule update --init --recursive
pip install -e third_party/linkerhand-python-sdk
pip install -e third_party/somehand
scripts/setup/download_somehand_l6_assets.sh
```

These packages are only required when `hands.enabled=true`.

### Sim2Real Recording

```bash
pip install -e '.[recording]'
```

Adds the Pico sim2real stack plus the video dependencies used by
`sim2real_record.yaml`. RealSense Python bindings are platform-specific: install
`pyrealsense2` manually in the active environment when using
`input.video.source=realsense`. On Arm machines, use conda-forge rather than the
pip package:

```bash
conda install -c conda-forge pyrealsense2
```

## Verify Installation

```bash
python -c "import teleopit; print('teleopit OK')"
python -c "import train_mimic.tasks; print('training OK')"  # if training installed
```

## Next Steps

- [Download Assets](download-assets) - Download models and data
- [Quick Start](quick-start) - Run your first simulation
