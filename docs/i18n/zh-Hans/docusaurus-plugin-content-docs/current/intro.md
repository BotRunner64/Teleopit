---
sidebar_position: 1
slug: /
---

# 简介

**Teleopit** 是一个轻量、可扩展的人形机器人全身遥操作框架。它能够将人类操作者的动作实时映射到 Unitree G1 机器人上，同时支持 MuJoCo 仿真和实物硬件部署。

## 核心特性

- **离线 sim2sim**：在 MuJoCo 中回放 BVH 动捕文件，通过 RL 策略驱动机器人
- **VR 遥操作**：基于 Pico 4 / Pico 4 Ultra 全身追踪的实时全身控制
- **Sim2Real 部署**：使用同一套流程直接部署到 Unitree G1 实物
- **机载控制**：控制回路可直接运行在 G1 的机载 NX 计算机上
- **训练流程**：基于 General-Tracking-G1 任务的端到端强化学习训练
- **可扩展设计**：基于协议的组件体系（InputProvider、Retargeter、Controller、Robot）

## 流程概览

```text
InputProvider (BVH / Pico4 VR)
    -> Retargeter (GMR)
    -> ObservationBuilder (166D)
    -> Controller (双输入 TemporalCNN ONNX)
    -> Robot (MuJoCo 仿真 或 Unitree G1)
```

## 技术规格

| 项目 | 参数 |
|------|------|
| 策略频率 | 50 Hz |
| PD 控制频率 | 1000 Hz |
| 观测维度 | 166D |
| 动作维度 | 29D（G1 关节） |
| ONNX 模型 | 双输入 TemporalCNN |
| 运动重定向 | GMR（General Motion Retargeting） |
| 仿真器 | MuJoCo |
| 硬件平台 | Unitree G1（29 自由度） |

## 下一步

- [安装指南](getting-started/installation) - 搭建开发环境
- [快速上手](getting-started/quick-start) - 运行你的第一个 sim2sim 示例
- [教程](tutorials/offline-sim2sim) - 各使用场景的详细步骤指引
