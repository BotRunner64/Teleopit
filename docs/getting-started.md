# 快速上手

这篇文档只回答一个问题：**你现在想做什么？**

如果你需要完整背景，先看 [`../README.md`](../README.md)。当前仓库默认主线仍是 `Tracking-Flat-G1-VelCmdHistory` 训练产出的 166D 双输入 ONNX，同时保留 `Tracking-Flat-G1-VelCmdHistoryAdaptive` 作为自适应采样训练变体。

## 先决条件

- Python `3.10+`
- 已执行 `pip install -e .`
- 已准备好 VelCmdHistory ONNX policy

如需训练或真机：

```bash
pip install -e '.[train]'
pip install -e '.[sim2real]'
```

## 选择你的目标

### 1. 我想离线跑一个 BVH

```bash
python scripts/run_sim.py           controller.policy_path=policy.onnx           input.bvh_file=data/lafan1/dance1_subject2.bvh
```

`hc_mocap` 示例：

```bash
python scripts/run_sim.py           controller.policy_path=policy.onnx           input.bvh_file=data/hc_mocap/walk.bvh           input.bvh_format=hc_mocap
```

继续阅读：[`inference.md`](inference.md)

### 2. 我想接入 UDP 实时动捕

```bash
# Terminal 1
python scripts/run_sim.py --config-name online controller.policy_path=policy.onnx

# Terminal 2
python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
```

继续阅读：[`inference.md`](inference.md)

### 3. 我想理解配置怎么改

常改项：

- `policy_hz`
- `pd_hz`
- `viewers`
- `input.bvh_file`
- `input.bvh_format`
- `controller.policy_path`

继续阅读：[`configuration.md`](configuration.md)

### 4. 我想准备训练数据

```bash
python train_mimic/scripts/data/build_dataset.py           --spec train_mimic/configs/datasets/twist2_full.yaml
```

如果只想先把原始数据转成标准 clips：

```bash
python train_mimic/scripts/data/ingest_motion.py           --type bvh           --input data/hc_mocap_bvh           --output data/datasets/hc_mocap_v1/clips/hc_mocap_v1           --source hc_mocap_v1           --bvh_format hc_mocap
```

继续阅读：[`dataset.md`](dataset.md)

### 5. 我想训练 / 导出 ONNX policy

```bash
python train_mimic/scripts/train.py           --motion_file data/datasets/twist2_full/train.npz
```

```bash
python train_mimic/scripts/save_onnx.py           --checkpoint logs/rsl_rl/g1_tracking_velcmd_history/<run>/model_30000.pt           --output policy.onnx           --history_length 10
```

继续阅读：[`training.md`](training.md)

### 6. 我想上真机 G1

```bash
python scripts/run_sim2real.py controller.policy_path=policy.onnx
```

继续阅读：[`sim2real.md`](sim2real.md)

## 常见误区

- 没有 policy 就直接运行：`controller.policy_path` 是必填项。
- 把旧 TWIST2、单输入或非 166D ONNX 拿来跑：当前主线只支持 VelCmdHistory 双输入 ONNX。
- 依赖默认 BVH 路径：请显式设置 `input.bvh_file`。
- 把训练、推理、sim2real 混成一个安装步骤：按需要安装 extras，排错更简单。
