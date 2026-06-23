<p align="center">
  <img src="assets/teleopit_logo.jpg" width="80" alt="Teleopit">
</p>

<h1 align="center">Teleopit</h1>

<p align="center">
  Lightweight, extensible whole-body teleoperation framework for humanoid robots.
  <br/>
  Real-time motion retargeting from BVH / Pico 4 VR to Unitree G1, in MuJoCo sim or on real hardware.
</p>

<p align="center">
  <a href="https://BotRunner64.github.io/Teleopit/">Documentation</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/zh-Hans/">中文文档</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/tutorials/pico-sim2sim">Pico Sim2Sim</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/tutorials/pico-sim2real">Pico Sim2Real</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/tutorials/training">Training</a>
</p>

---

## Quick Start — Minimal Sim2Sim

**1. Install**

```bash
pip install -e .
```

**2. Download assets**

```bash
pip install modelscope
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
```

The canonical Unitree G1 robot model is downloaded to
`assets/robots/unitree_g1/g1_29dof.xml`. Training, sim2sim, retargeting, and FK
validation all use this same XML.

**3. Run**

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

You should see a MuJoCo viewer with the robot tracking the BVH motion.

To show the simulated D435i RGB camera view, add the explicit `camera` viewer:

```bash
python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh \
    'viewers=[sim2sim,camera]'
```

For sim2real, viewers are disabled by default. Add `viewers=retarget` to show
the retargeted reference in an optional MuJoCo window.

## Pico Motion Recording

Record many Pico clips as training-ready G1 motion NPZ files:

```bash
pip install -e '.[pico4]'
python scripts/run/record_pico_motion.py
```

The recorder starts the Pico receiver and live Retarget viewer before waiting
for clip names, so preview keeps running while the terminal is idle. Enter a
semantic clip name, then use `R` to start, `S` to save, `D` to discard, `N` for
a new name, and `Q` to quit. Saved clips are written to
`data/pico_motion/clips/` using the semantic label in the filename, with no
sidecar JSON.

Merge recorded clips into the standard HDF5 shard dataset:

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec data/pico_motion/pico_recorded.yaml --force
```

## Sim2Real HDF5 Recording

Pico sim2real can also record manual HDF5 episodes from the real G1:

```bash
pip install -e '.[recording]'
# If you use RealSense video, install pyrealsense2 manually for your platform.
# On Arm machines, prefer conda-forge:
# conda install -c conda-forge pyrealsense2
python scripts/run/run_sim2real.py --config-name sim2real_record \
    controller.policy_path=track.onnx \
    recording.task="walk forward"
```

Recording uses the terminal controls `R` start, `S` save, `D` discard, and `Q`
shutdown. `STANDING`, `MOCAP`, `ARMS`, and paused mocap can be recorded. Saved
episodes are written as `.h5` files under `data/recordings/sim2real_hdf5/episodes/`.
`sim2real_record.yaml` stores camera frames as compressed MP4 sidecar files under
`data/recordings/sim2real_hdf5/videos/` and keeps `frame_index` / `timestamp`
sync metadata in the HDF5 episode. The low-dimensional HDF5 schema records
`observation.state(68)`, `observation.mode(1)`, `action(36)` as the aligned
reference qpos sent to the policy path, and `action.hand(12)` as the latest
LinkerHand left/right 6D pose commands.

## Documentation

Full docs at **[BotRunner64.github.io/Teleopit](https://BotRunner64.github.io/Teleopit/)**, covering installation profiles, all tutorials, configuration reference, and architecture.

## Changelog

### Unreleased

- Added Pico sim2real `ARMS` mode: Pico/controller `B` toggles between whole-body `MOCAP` and stand-pose body/legs with live retargeted arms.
- Bumped Pico input support to pico-bridge 0.2.1 and its corrected tracking pose semantics.
- Added optional LinkerHand L6 sim2real modes under `hands.*`: `gripper` from Pico grip/trigger and `vr_hand_pose` from Pico hand pose through somehand 0.2.0 public API.
- Added manual Pico sim2real HDF5 recording with RealSense D435i RGB video, 68D robot state, mode labels, 36D reference-qpos action labels, and 12D LinkerHand pose action labels.
- Added LinkerHand O6 support for Pico `gripper` mode with an O6-specific grasp pose.
- Set LinkerHand L6 `vr_hand_pose` control to maximum speed while keeping `gripper` at the configured default speed.
- Switched default `vr_hand_pose` to a low-latency somehand path with 60 Hz hand retargeting and reduced smoothing.
- Realtime mode switches and pause/resume now preserve GMR IK warm-starts instead of cold-starting the retargeter on each transition.
- Added an interactive Pico motion recorder that saves retargeted G1 motion clips as training-ready NPZ files.
- Switched dataset build outputs to recursive minimal HDF5 shards with no train/val split or manifest; `precompute_dataset.py` turns them into separate precomputed training datasets before training.
- General-Tracking-G1 training defaults to `rewind` motion sampling and also supports `uniform`; playback/benchmark use `start`.
- Added optional `sampling_mode=rewind` for training, which restarts failed episodes from the same clip after rewinding a configurable number of policy steps.
- Added root velocity, joint tracking, and survival rewards to the General-Tracking-G1 training objective.
- Renamed General-Tracking-G1 observation terms to explicit `ref_*`, `robot_*`, and `prev_action` keys.

### v0.3.0 (2026-05-12)

- Consolidated realtime input around pico-bridge 0.2.0 and removed the old ZMQ/onboard Pico path.
- Unified sim/sim2real reference buffering, resume realignment, and velocity smoothing.
- Added UDP BVH realtime input, online sim config, multi-viewer support, and fixed camera viewing.
- Split sim2real reference/safety runtime modules and updated the G1 MuJoCo camera asset.

### v0.2.0 (2026-04-03)

- Added Pico 4 teleoperation through pico-bridge and the G1 Bridge SDK.
- Added offline playback keyboard controls, Pico sim2sim mode control, and a standalone standing controller.
- Improved realtime mocap buffering/catch-up and upgraded the released model to the 30k checkpoint.

### v0.1.1 (2025-03-28)

- Dataset shard-only refactor
- External asset management (ModelScope), repository slimming

### v0.1.0 (2025-03-25)

- Initial public release: General-Tracking-G1 training, ONNX sim2sim inference, Pico 4 VR teleoperation, Unitree G1 hardware deployment

## License

[Apache 2.0](LICENSE)
