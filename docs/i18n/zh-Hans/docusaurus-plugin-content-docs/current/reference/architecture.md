---
sidebar_position: 1
---

# 架构

面向开发者的系统内部结构和技术约束。

## Pipeline

```text
InputProvider（BVH 文件 / Pico4）
    -> Retargeter（GMR）
    -> ObservationBuilder（167D）
    -> Controller（双输入 TemporalCNN ONNX）
    -> Robot（MuJoCo 仿真或 Unitree G1）
```

离线/在线推理由 `teleopit/runtime/` 和 `teleopit/pipeline.py` 装配。硬件状态机通过 `teleopit/sim2real/mp/` 中的进程隔离运行时执行。训练由 `train_mimic/` 提供。

## 代码结构

```text
configs / scripts
    -> runtime
    -> interfaces + pipeline state machines
    -> adapters（inputs / retargeting / controller / robot / recording）

train_mimic/scripts
    -> train_mimic/app.py
    -> single task registry / env builder / runner cfg
    -> mjlab / rsl_rl

train_mimic/scripts/data
    -> train_mimic/data/dataset_builder.py
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

## 核心模块边界

| 模块 | 职责 |
|------|------|
| `teleopit/interfaces.py` | 稳定协议：InputProvider、Retargeter、Controller、Robot、ObservationBuilder、Recorder |
| `teleopit/runtime/` | 配置解析、路径规范化、组件装配、CLI 校验 |
| `teleopit/pipeline.py` | 离线仿真的轻量 facade |
| `teleopit/sim2real/mp/` | 进程隔离的 sim2real 状态机、IPC 和机器人控制循环 |
| `teleopit/controllers/observation.py` | ObservationBuilder |
| `teleopit/controllers/rl_policy.py` | 接受观测维度与运行时 builder 匹配的双输入 ONNX |
| `train_mimic/app.py` | 共享的训练/播放/benchmark 装配 |
| `train_mimic/tasks/tracking/config/` | 单一任务注册（`General-Tracking-G1`） |
| `train_mimic/data/dataset_builder.py` | 唯一官方数据集构建入口 |

## 技术规格

| 项目 | 规格 |
|---|---|
| 训练任务 | `General-Tracking-G1` |
| 推理观测 | `velcmd_history`（167D） |
| ONNX 签名 | 双输入 `obs`（167D）+ `obs_history` |
| Actor/Critic | TemporalCNN（2048、1024、512、256、128） |
| 训练采样 | 默认 `rewind`；也支持 `uniform`；播放/评估使用 `start` |
| 训练 `window_steps` | `[0]` |
| 数据格式 | 可递归发现的最小 HDF5 shard（`shard_*.h5`） |

## 约束

- 必须显式提供 `controller.policy_path`，且文件必须存在
- 离线 BVH 运行必须显式提供 `input.bvh_file`
- `viewers` 是唯一的 viewer 配置入口
- 观测/ONNX 维度不匹配会在启动时立即报错
- sim2real 也要求双输入 ONNX，且观测维度必须与运行时 builder 匹配

## 公共接口

**稳定运行模式：** 离线 sim2sim、离线 sim2real playback、Pico4 sim2sim、G1 sim2real

**稳定训练入口：** `train.py`、`play.py`、`benchmark.py`、`save_onnx.py`

**稳定数据入口：** `build_dataset.py`
