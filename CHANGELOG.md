# Changelog

## [0.1.0] - 2025-03-25

### Initial Release

- **Training**: `General-Tracking-G1` task — TemporalCNN actor/critic (1024-512-256-256-128), 166D VelCmd observation, mjlab 1.2.0 + rsl_rl PPO, 30k iterations
- **Sim2Sim**: MuJoCo simulation with ONNX policy inference (50Hz policy, 200Hz PD, Newton solver, implicitfast integrator)
- **Sim2Real**: Unitree G1 real robot control via CycloneDDS (UDP BVH + Pico4 VR input)
- **Motion Retargeting**: GMR-based retargeting pipeline supporting BVH, UDP, and Pico4 input sources
- **Dataset**: SEED / LAFAN1 / TWIST2 motion dataset processing and sampling pipeline
- **Tooling**: Benchmark evaluation, ONNX export, dataset review utilities, multi-GPU training launcher
