# Changelog

## [0.3.0] - 2026-05-12

- 重构实时输入栈，Pico 4 统一使用 pico-bridge 0.2.0 in-process receiver，并移除旧 ZMQ/onboard Pico 路径。
- 统一 sim/sim2real 实时 reference buffer、pause/resume realignment 与速度平滑逻辑。
- 扩展 UDP BVH、online sim、多 viewer 与固定相机支持。
- 拆分 sim2real reference/safety 运行时模块，并更新 G1 MuJoCo 相机资产。

## [0.2.0] - 2026-04-03

- 接入 Pico 4 遥操作与 G1 Bridge SDK。
- 新增独立 Standing 控制器、离线播放键盘控制与 Pico sim2sim 模式控制。
- 优化实时 mocap 缓冲与 catch-up，并将发布模型升级至 30k checkpoint。

## [0.1.1] - 2025-03-28

数据集 shard-only 改造、adaptive_bin 采样、外部资源管理、仓库瘦身。

## [0.1.0] - 2025-03-25

首个公开版本：General-Tracking-G1 全身追踪训练、ONNX sim2sim 推理、Pico 4 VR 遥操作、Unitree G1 真机部署。
