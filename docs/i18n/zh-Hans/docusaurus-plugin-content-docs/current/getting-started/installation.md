---
sidebar_position: 1
---

# 安装

Teleopit 提供多种安装配置，你可以根据实际使用场景选择对应的安装方式。

## 前置条件

- Python 3.10+
- [Conda](https://docs.conda.io/)（推荐）

```bash
conda create -n teleopit python=3.10
conda activate teleopit
```

## 安装配置

### 仅推理（sim2sim）

```bash
pip install -e .
```

该配置已足够进行离线 BVH 回放和 MuJoCo 仿真。

### 训练

```bash
pip install -e '.[train]'
```

额外安装 `rsl-rl-lib`、`mjlab`、`wandb` 等训练相关依赖。

### Sim2Real（硬件部署）

```bash
pip install -e '.[sim2real]'
```

额外安装 `opencv-python` 和 `g1_bridge_sdk`。此外还需要初始化子模块并编译 C++ 桥接库：

```bash
git submodule update --init --recursive
bash scripts/setup/setup_g1_bridge.sh
```

详见 [G1 Bridge SDK](../reference/g1-bridge-sdk)。

### Pico 4 VR

```bash
pip install -e '.[pico4]'
```

Teleopit 使用进程内的 `pico_bridge.PicoBridge` receiver 接收 Pico 追踪数据。
receiver 可以运行在工作站 PC，也可以运行在机器人 onboard 计算机。
完整设置流程详见 [Pico Sim2Sim](../tutorials/pico-sim2sim) 和
[Pico Sim2Real](../tutorials/pico-sim2real)。

## 验证安装

```bash
python -c "import teleopit; print('teleopit OK')"
python -c "import train_mimic.tasks; print('training OK')"  # 仅在安装了训练配置时适用
```

## 下一步

- [下载资源](download-assets) - 下载模型和数据
- [快速上手](quick-start) - 运行你的第一个仿真
