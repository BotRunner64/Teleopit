---
sidebar_position: 3
---

# Sim2Real 部署

通过 g1_bridge_sdk（C++ DDS 桥接）将 Teleopit 部署到 Unitree G1 真实机器人上。

:::tip
如果使用 Pico 4 / Pico 4 Ultra VR 部署，请参阅完整的 [Pico VR 教程](pico4-vr)。
:::

## 输入源

| 输入源 | 配置 | 说明 |
|--------|------|------|
| **Pico 4 / Pico 4 Ultra** | `--config-name pico4_sim2real` | [Pico VR](pico4-vr) |
| 离线 BVH 文件 | 默认配置 | 本页内容 |

## 前置要求

**硬件：**
- Unitree G1（29 自由度）
- Unitree 无线遥控器
- 机器人与控制 PC 之间的网络连接
- Pico 4 头显，或离线 BVH 动捕文件

**软件：**
```bash
pip install pybind11
pip install third_party/g1_bridge_sdk
pip install -e '.[sim2real]'
```

使用 Pico 路线时，还需执行：`bash scripts/setup/setup_pico4.sh`

## 离线 BVH 播放

```bash
python scripts/run/run_sim2real.py \
    controller.policy_path=track.onnx \
    real_robot.network_interface=eth0 \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

### 遥控器按键映射

| 按键 | 功能 |
|------|------|
| `Start` | 进入 `STANDING` |
| `Y` | 开始播放 / 进入 `MOCAP` |
| `A` | 暂停 / 恢复 |
| `B` | 从头重播 |
| `X` | 返回 `STANDING` |
| `L1+R1` | 急停（`DAMPING`） |

## 控制模式

| 模式 | 数据流 | 使用场景 |
|------|--------|---------|
| `STANDING` | 默认姿态 -> RL 策略 -> 关节 | 启动、恢复、等待 |
| `MOCAP` | Pico/BVH -> 重定向 -> RL 策略 -> 关节 | 遥操作 / 动作回放 |
| `DAMPING` | 发送阻尼指令 | 急停 |

### 状态机

```text
                     +-----------------------------+
                     |    L1+R1 急停（任意状态）      |
                     v                             |
  [IDLE] --Start--> [STANDING] --Y--> [MOCAP] --X--> [STANDING]
                           ^                       |
                           +----------Y------------+
    ^                                                  |
    +------------------Start---------------------------+
                           [DAMPING]
```

### Pico MOCAP 子状态

当使用 `input.provider=pico4` 时，MOCAP 模式包含以下子状态：

- **ACTIVE**：正常的实时动捕追踪
- **PAUSED**：冻结参考姿态；机器人保持平衡但停止跟随
- **RESUMING**：清除实时参考缓冲区，重建偏航/轴心对齐，平滑过渡回实时动捕

## 常用参数

```bash
# 调整控制频率
policy_hz=30

# 指定 BVH 文件
input.bvh_file=data/sample_bvh/aiming1_subject1.bvh

# Pico 超时时间
input.pico4_timeout=30

# 暂停/恢复过渡参数
pause_resume_transition_duration=1.5
pause_resume_warmup_steps=3

# 网络接口
real_robot.network_interface=enp3s0
```

## 独立站立测试

一个用于快速验证硬件和策略的最小脚本，不依赖主框架：

```bash
python scripts/run/standalone_standing.py \
    --policy track.onnx \
    --network-interface eth0
```

支持 `--dry-run` 模式，在不发送电机指令的情况下进行安全的时序基准测试。
