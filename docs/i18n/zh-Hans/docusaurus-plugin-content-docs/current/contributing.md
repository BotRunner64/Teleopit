---
sidebar_position: 100
---

# 贡献指南

Teleopit 的开发与扩展指南。

## 开发环境搭建

```bash
git clone https://github.com/BotRunner64/Teleopit.git
cd Teleopit
pip install -e '.[dev]'
```

## 运行测试

```bash
pytest tests/
```

测试套件覆盖单元测试、集成测试和端到端 pipeline 测试。

## 项目结构

```text
teleopit/           # 核心推理与部署包
├── interfaces.py   # Protocol 定义
├── pipeline.py     # TeleopPipeline 门面
├── controllers/    # RL 策略控制器 + 观测构建器
├── robots/         # MuJoCoRobot 实现
├── inputs/         # BVHInputProvider、Pico4InputProvider、UDPBVHInputProvider
├── retargeting/    # GMR 动作重定向
├── sim/            # SimulationLoop、参考运动工具
├── sim2real/       # 真机状态机
├── recording/      # Pico motion 录制辅助工具
├── runtime/        # 配置解析、工厂、外部资源管理
├── bus/            # InProcessBus 进程内通信
└── configs/        # Hydra YAML 配置

train_mimic/        # 训练 pipeline 与 RL 框架
├── tasks/tracking/ # General-Tracking-G1 训练任务
├── scripts/        # 训练、评估、ONNX 导出
└── data/           # 数据集构建工具

scripts/            # 入口脚本
├── run/            # run_sim.py、run_sim2real.py
├── setup/          # 资源下载与环境配置
├── dev/            # 开发工具
└── render/         # 视频渲染

tests/              # 测试套件
```

## 模块隔离

所有模块通过 `InProcessBus`（零拷贝设计）进行通信。添加新组件时请遵循现有模式：

1. 在 `interfaces.py` 中定义 Protocol
2. 实现 Adapter
3. 在 `runtime/factory.py` 中注册
4. 在 `configs/` 中添加配置

## 提交前检查

1. 运行测试：`pytest tests/`
2. 检查大文件：`python scripts/dev/check_large_tracked_files.py`
3. 确认 fail-fast 原则：不做静默 pad、clip 或默认回退
