# Changelog

## [0.4.0] - 2026-06-25

- 改进 Pico 实时控制：支持 pico-bridge 0.2.1、`ARMS` 模式，以及保留 retargeter warm-start 的模式切换/暂停恢复。
- 新增可选 LinkerHand L6/O6 sim2real 控制，支持 Pico gripper 输入和低延迟 L6 `vr_hand_pose`。
- 新增 Pico sim2real 手动 HDF5 录制，以及用于训练数据采集的交互式 Pico motion recorder。
- 优化训练数据流程：minimal HDF5 shards、显式 precompute、rewind 采样和更新后的 tracking rewards。

## [0.3.0] - 2026-05-12

- 重构实时输入栈，Pico 4 统一使用 pico-bridge 0.2.0 in-process receiver，并移除旧 ZMQ/onboard Pico 路径。
- 统一 sim/sim2real 实时 reference buffer、pause/resume realignment 与速度平滑逻辑。
- 扩展 UDP BVH、online sim、多 viewer 与固定相机支持。
- 拆分 sim2real reference/safety 运行时模块，并更新 G1 MuJoCo 相机资产。

## [0.2.0] - 2026-04-03

- 接入 Pico 4 遥操作与 G1 Bridge SDK。
- 新增独立 Standing 控制器、离线播放键盘控制与 Pico sim2sim 模式控制。
- 优化实时 mocap 缓冲与 catch-up，并将发布模型升级至 30k checkpoint。

## [0.1.1] - 2026-03-28

- 数据集改为 shard-only 输出。
- 引入外部资源管理并瘦身仓库。

## [0.1.0] - 2026-03-25

- 首个公开版本。
- 支持 General-Tracking-G1 全身追踪训练与 ONNX sim2sim 推理。
- 支持 Pico 4 VR 遥操作与 Unitree G1 真机部署。
