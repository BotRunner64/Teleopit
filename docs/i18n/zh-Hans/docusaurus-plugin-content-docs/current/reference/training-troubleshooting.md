---
sidebar_position: 5
---

# 训练问题排查

常见训练问题及解决方案。

:::info
训练流程见[训练教程](../tutorials/training)，数据准备见[数据集参考](dataset)。
:::

---

## 问题 1：Mean Episode Length = 1.00（机器人第一步就终止）

### 现象

- `Mean episode length: 1.00`
- `Episode_Termination/anchor_pos` 接近并行环境总数
- `Metrics/motion/error_anchor_pos` > 0.5 m
- `Metrics/motion/error_body_rot` 很大（接近 pi）

### 根本原因

通常不是 PPO 超参数问题，而是 **motion NPZ 的监督标签与 MuJoCo FK 不一致**：

1. **body 位置坐标系错误**：把局部坐标当世界坐标使用
2. **body 顺序错误**：使用 PKL 的 38-body 顺序而非 mjlab G1 的 30-body 顺序
3. **body 朝向/角速度标签错误**：所有 body 近似为 root 朝向

当前版本的 `convert_pkl_to_npz.py` 已修复上述问题。

### 快速排查

```bash
python train_mimic/scripts/data/check_motion_npz_fk.py \
    --npz data/lafan1_clips/lafan1/<clip>.npz
```

推荐判据：`pos_max < 1e-3 m`、`quat_mean < 0.05 rad`、`quat_p95 < 0.10 rad`。

如果检查失败，重新生成数据并做一次 smoke test：

```bash
python train_mimic/scripts/train.py \
    --num_envs 64 --max_iterations 100 \
    --motion_file data/datasets/<dataset>_precomputed
```

预期：`Mean episode length` 明显大于 1，`error_anchor_pos` 开始下降。

---

## 问题 2：Episode Length 不增长

### 现象

训练 1000+ 轮后，`Mean episode length` 仍然很低（< 3），无上升趋势。

### 可能原因

1. Retargeting 质量差（目标姿态不可达）
2. Tracking reward 权重过低，正则化权重过高
3. 学习率过大/过小，clip_param 不匹配
4. 终止条件过严

### 排查步骤

1. 用 `play.py` 可视化参考运动
2. 检查奖励分布——tracking reward 应占主导
3. 临时增大 `bad_anchor_pos` 阈值（0.25m → 0.5m）
4. 对比 mjlab 内置的 G1 tracking task

---

## 问题 3：训练速度慢

### 现象

训练速度 < 1000 steps/s（RTX 4090 预期 1500-2000 steps/s）。

### 解决方案

1. 增加 `--num_envs` 到 4096（需要 24 GB 显存）
2. 训练时关闭 `--video`
3. 使用 TensorBoard 替代 W&B（默认即 TensorBoard）

---

## 问题 4：`nefc overflow - please increase njmax`

### 现象

```text
nefc overflow - please increase njmax to 257
```

### 根本原因

MuJoCo 约束缓冲区不足。机器人跌倒或大量接触时，活跃约束数超出 `njmax`。`mjlab` 训练默认 `sim.njmax=250`。

### 解决方案

仓库中已修复。`train_mimic/tasks/tracking/config/env.py` 的 env builder 覆盖了训练仿真参数：

```python
self.sim.njmax = 500
self.sim.nconmax = 150_000
```

如果警告仍出现在更高数值，增加到 `njmax = 800`。

:::note
仅修改机器人 XML 不够——`mjlab` 的仿真层 `njmax` 才是实际生效的参数。
:::

---

## 问题 5：Benchmark 视频问题

### 视频只有 1 帧

确保 `num_eval_steps >= video_length`：

```bash
python train_mimic/scripts/benchmark.py \
    --checkpoint logs/rsl_rl/g1_general_tracking/<run>/model_30000.pt \
    --motion_file data/datasets/<dataset>_precomputed \
    --num_envs 1 --num_eval_steps 2000 \
    --video --video_length 600
```

### EGL/OpenGL 错误

安装 OpenGL/EGL 依赖：

```bash
conda install -c conda-forge libopengl libglx libegl libglvnd pyopengl
```

如果 GPU EGL 不可用，尝试 CPU 渲染：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    python train_mimic/scripts/benchmark.py ... --video
```

---

## 问题 6：Sim2Sim 脚滑（Benchmark 正常但 ONNX 推理脚打滑）

### 根本原因

sim2sim 配置参数与训练环境不一致：

1. **`default_angles` 不匹配（最关键）**：不同的关节默认值导致动作偏移和观测误差
2. **缺少 joint armature**：训练环境有非零 armature，零 armature 导致过冲
3. **condim 不一致**：训练和 sim2sim 之间碰撞参数不同

### 诊断方法

```python
from mjlab.asset_zoo.robots import get_g1_robot_cfg
cfg = get_g1_robot_cfg()
print(cfg.init_state.joint_pos)  # 必须与 g1.yaml default_angles 一致
```

### 解决方案

更新 `teleopit/configs/robot/g1.yaml` 和 `assets/robots/unitree_g1/g1_29dof.xml`，使其与训练环境的值一致（default angles、armature、condim）。

此修复同时影响 sim2real 路径，因为 `default_angles` 被 `rl_policy.py` 和 `observation.py` 共用。
