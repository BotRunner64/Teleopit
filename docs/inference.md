# 推理与运行指南

当前推理主线已经收敛为单一路径：**166D VelCmdHistory 观测 + 双输入 ONNX policy（`obs` + `obs_history`）**。

这篇文档覆盖离线 sim2sim、UDP 实时 online sim2sim、viewer、录制和离线渲染。

## 运行前确认

- 已安装核心依赖：`pip install -e .`
- 已准备由 `train_mimic/scripts/save_onnx.py` 导出的 VelCmdHistory ONNX
- `controller.policy_path=...` 已显式提供
- 离线 BVH 运行时请显式传 `input.bvh_file=...`

## 运行约束

- 运行时只接受 166D VelCmdHistory 双输入 ONNX。
- 单输入 ONNX、旧 TWIST2 ONNX 和其他观测维度会在启动时直接报错。
- 观测定义与 ONNX 输入维度不匹配时 fail fast，不会自动 pad/trim。
- `viewers` 是唯一 viewer 配置入口；旧 `viewer` alias 已移除。

## 离线 sim2sim

```bash
python scripts/run_sim.py           controller.policy_path=policy.onnx           input.bvh_file=data/lafan1/dance1_subject2.bvh
```

`hc_mocap` 示例：

```bash
python scripts/run_sim.py           controller.policy_path=policy.onnx           input.bvh_file=data/hc_mocap/walk.bvh           input.bvh_format=hc_mocap
```

常用 override：

```bash
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=none
python scripts/run_sim.py controller.policy_path=policy.onnx viewers=all
python scripts/run_sim.py controller.policy_path=policy.onnx 'viewers=[retarget,sim2sim]'
python scripts/run_sim.py controller.policy_path=policy.onnx +num_steps=5000 +record=true
```

## UDP 实时 online sim2sim

```bash
# Terminal 1
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

常见 override：

```bash
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx input.udp_port=1119
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx viewers=none
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx num_steps=300
```

运行时行为：

- `UDPBVHInputProvider` 始终返回最新帧，不维护内部播放计数器。
- `fps=30` 固定，`SimulationLoop` 按 `bvh_idx = int(policy_time × input_fps)` 做时间对齐。
- `num_steps=0` 表示无限循环。
- `realtime=true` 时即使没有 viewer 也会做 wall-clock 限速。

## Viewer 模式

`viewers` 支持：

- `sim2sim`
- `retarget`
- `bvh`
- `all`
- `none`

注意事项：

- 多 viewer 运行在独立子进程中。
- Hydra list override 需要 shell 引号，例如 `'viewers=[retarget,sim2sim]'`。
- 当所有活动 viewer 被关闭时，仿真会自动结束。

## 录制

```bash
python scripts/run_sim.py           controller.policy_path=policy.onnx           input.bvh_file=data/lafan1/dance1_subject2.bvh           +record=true           recording.output_path=outputs/session.h5
```

录制文件包含：

- `joint_pos`
- `joint_vel`
- `mimic_obs`
- `action`
- `target_dof_pos`
- `torque`
- `timestamp`

## 渲染

```bash
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/lafan1/dance1_subject2.bvh --policy policy.onnx
MUJOCO_GL=egl python scripts/render_sim.py --bvh data/hc_mocap/wander.bvh --format hc_mocap --policy policy.onnx
```

## 继续阅读

- 配置说明：[`configuration.md`](configuration.md)
- 数据准备：[`dataset.md`](dataset.md)
- 训练与导出：[`training.md`](training.md)
- 真机部署：[`sim2real.md`](sim2real.md)
