# 架构与边界

本文档面向开发者，描述系统内部结构和技术约束。

## 运行主线

```text
InputProvider (BVH file / UDP realtime / Pico4)
    -> Retargeter (GMR)
    -> ObservationBuilder (166D)
    -> Controller (dual-input TemporalCNN ONNX)
    -> Robot (MuJoCo sim or Unitree G1)
```

离线/在线推理由 `teleopit/runtime/` 和 `teleopit/pipeline.py` 装配，真机状态机保留在 `teleopit/sim2real/controller.py`。训练闭环由 `train_mimic/` 提供。

## 代码结构

```text
configs / scripts
    -> runtime
    -> interfaces + pipeline state machines
    -> adapters (inputs / retargeting / controller / robot / recording)

train_mimic/scripts
    -> train_mimic/app.py
    -> single task registry / env builder / runner cfg
    -> mjlab / rsl_rl

train_mimic/scripts/data
    -> train_mimic/data/dataset_builder.py
    -> dataset_lib / motion_fk / convert_pkl_to_npz
```

## 核心边界

- `teleopit/interfaces.py` — 稳定协议：`InputProvider`、`Retargeter`、`Controller`、`Robot`、`ObservationBuilder`、`Recorder`
- `teleopit/runtime/` — 配置解析、路径归一、组件装配、CLI 公共校验
- `teleopit/pipeline.py` — offline / online sim 的轻量 façade
- `teleopit/sim2real/controller.py` — 真机状态机与控制行为
- `teleopit/controllers/observation.py` — `ObservationBuilder`
- `teleopit/controllers/rl_policy.py` — 只接受 166D 双输入 ONNX
- `train_mimic/app.py` — train/play/benchmark 共用 task 装配与校验
- `train_mimic/tasks/tracking/config/` — 唯一 task 注册面（`General-Tracking-G1`）
- `train_mimic/data/dataset_builder.py` — 数据集构建唯一正式入口

## 技术规格

- 训练 task：`General-Tracking-G1`
- 推理观测：`velcmd_history`（166D）
- ONNX 签名：dual-input `obs`（166D）+ `obs_history`
- actor/critic：TemporalCNN（1024, 512, 256, 256, 128）
- 训练采样模式：`uniform`；playback / benchmark 采样模式：`start`
- 训练 `window_steps=[0]`
- 数据格式：支持单个 merged NPZ 或 shard 目录（多个 NPZ）

## Realtime 配置

在线 realtime 路径会先把 retarget 后的 `qpos` 写入短时 reference timeline，再按控制时刻从 timeline 采样 reference window。相关配置入口：

- `retarget_buffer_enabled` / `retarget_buffer_window_s` / `retarget_buffer_delay_s`
- `reference_steps`
- `realtime_buffer_warmup_steps` / `realtime_buffer_low_watermark_steps` / `realtime_buffer_high_watermark_steps`
- `reference_velocity_smoothing_alpha` / `reference_anchor_velocity_smoothing_alpha`

## 约束

- `controller.policy_path` 必须显式提供且文件必须存在
- 离线 BVH 运行必须显式提供 `input.bvh_file`
- `viewers` 是唯一 viewer 配置入口
- 观测定义与 ONNX 输入维度不匹配时启动即报错
- sim2real 也只支持 166D 双输入 ONNX

## 当前公共面

- 稳定运行模式：offline sim2sim、UDP online sim2sim、Pico4 sim2sim、G1 sim2real
- 稳定训练入口：`train.py`、`play.py`、`benchmark.py`、`save_onnx.py`
- 稳定数据入口：`build_dataset.py`、`split_shards.py`
