---
sidebar_position: 100
---

# Contributing

Guidelines for developing and extending Teleopit.

## Development Setup

```bash
git clone https://github.com/BotRunner64/Teleopit.git
cd Teleopit
pip install -e '.[dev]'
```

## Running Tests

```bash
pytest tests/
```

The test suite covers unit tests, integration tests, and end-to-end pipeline tests.

## Project Structure

```text
teleopit/           # Core inference & deployment package
├── interfaces.py   # Protocol definitions
├── pipeline.py     # TeleopPipeline facade
├── controllers/    # RL policy controller + observation builder
├── robots/         # MuJoCoRobot implementation
├── inputs/         # BVHInputProvider, Pico4InputProvider, UDPBVHInputProvider
├── retargeting/    # GMR motion retargeting
├── sim/            # SimulationLoop, reference motion utilities
├── sim2real/       # Hardware state machines
├── recording/      # HDF5Recorder
├── runtime/        # Config parsing, factories, external assets
├── bus/            # InProcessBus for inter-component communication
└── configs/        # Hydra YAML configurations

train_mimic/        # Training pipeline & RL framework
├── tasks/tracking/ # General-Tracking-G1 training task
├── scripts/        # Training, evaluation, ONNX export
└── data/           # Dataset construction tools

scripts/            # Entry points
├── run/            # run_sim.py, run_sim2real.py
├── setup/          # Asset download and setup
├── dev/            # Development utilities
└── render/         # Video rendering

tests/              # Comprehensive test suite
```

## Module Isolation

All modules communicate via `InProcessBus` (zero-copy design). Follow existing patterns when adding new components:

1. Define a protocol in `interfaces.py`
2. Implement the adapter
3. Register it in `runtime/factory.py`
4. Add configuration in `configs/`

## Pre-Push Checklist

1. Run tests: `pytest tests/`
2. Check for large files: `python scripts/dev/check_large_tracked_files.py`
3. Verify fail-fast: no silent padding, clipping, or default fallbacks
