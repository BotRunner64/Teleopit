---
sidebar_position: 1
---

# 架构

本页介绍 Teleopit 的整体架构设计、核心模块边界及技术规格。

## 运行主线 Pipeline

Teleopit 的运行主线是一条线性数据流管线：

```
输入源 (Input)
  → 重定向 (Retarget)
    → 观测构建 (Observation)
      → 策略推理 (Policy)
        → PD 控制 (Controller)
          → 仿真/实机执行 (Sim / Real)
```

每个环节职责单一，通过明确定义的接口相互连接。

## 代码结构

```
teleopit/
├── app.py                 # 应用入口
├── interfaces.py          # 核心接口定义
├── runtime/               # 运行时配置装配与启动逻辑
├── pipeline/              # 数据流管线
├── sim2real/              # 实机部署适配层
├── observation/           # 观测构建
├── rl_policy/             # 强化学习策略推理
├── task/                  # 任务配置
└── dataset_builder/       # 数据集构建工具
```

## 核心模块边界

| 模块 | 文件/目录 | 职责 |
|---|---|---|
| 接口层 | `interfaces.py` | 定义所有核心抽象接口，模块间仅通过接口通信 |
| 运行时 | `runtime/` | Hydra 配置加载、对象组装、依赖注入 |
| Pipeline | `pipeline/` | 数据流编排，驱动每一帧的采样-推理-执行循环 |
| Sim2Real | `sim2real/` | 实机通信适配（DDS 桥接、状态同步） |
| 观测 | `observation/` | 从仿真/实机状态构建策略所需的观测向量 |
| 策略 | `rl_policy/` | ONNX 模型加载与推理，action 后处理 |
| 入口 | `app.py` | 命令行入口，调用 runtime 装配并启动 pipeline |
| 任务配置 | `task/` | Hydra 配置文件（YAML） |
| 数据集 | `dataset_builder/` | 动捕数据转换、NPZ 打包、数据集分片 |

## 技术规格

| 项目 | 规格 |
|---|---|
| 仿真引擎 | MuJoCo |
| 策略推理 | ONNX Runtime |
| 配置系统 | Hydra |
| 实机通信 | DDS（Cyclone DDS） |
| 支持的机器人 | 宇树 G1 |
| 输入源 | BVH 文件、Pico 4 VR 头显 |

## 约束

- **单机器人**：当前架构假设同一时刻只控制一台机器人。
- **固定观测格式**：观测构建器的输出维度在初始化时确定，运行时不可变。
- **同步 pipeline**：pipeline 各阶段串行执行，策略推理频率即为 pipeline 的帧率。
- **ONNX 模型**：策略必须导出为 ONNX 格式，不直接支持 PyTorch checkpoint。

## 公共接口

以下接口构成 Teleopit 的公共 API 面，外部扩展应仅依赖这些接口：

- `interfaces.py` 中定义的抽象基类（InputProvider、ObsBuilder、Controller 等）
- `runtime/` 提供的工厂注册机制
- Hydra 配置 schema

内部实现细节（如 pipeline 的具体调度逻辑）不属于公共 API，可能在版本迭代中变更。
