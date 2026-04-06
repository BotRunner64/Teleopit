---
sidebar_position: 3
---

# 配置常见问题

## 为什么设置了 `policy_path` 还是启动不了？

1. 确认文件存在
2. 确认不是旧的 1402D / TWIST2 ONNX 模型
3. 确认输入维度是 `166`，且为双输入 ONNX（`obs` + `obs_history`）

## 为什么必须显式指定 `input.bvh_file`？

`input/bvh.yaml` 已不再提供机器相关的默认路径。始终在命令行显式指定：

```bash
python scripts/run/run_sim.py \
    controller.policy_path=policy.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

## 为什么 `viewer=true` 不起作用？

旧的 `viewer` 别名已移除。请使用 `viewers`（复数）：

```bash
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=sim2sim
python scripts/run/run_sim.py controller.policy_path=policy.onnx viewers=none
```
