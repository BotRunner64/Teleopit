# 架构与边界

Teleopit 当前按两条运行主线组织：训练侧支持 `Tracking-Flat-G1-VelCmdHistory`、`Tracking-Flat-G1-VelCmdHistoryAdaptive` 和 `Tracking-Flat-G1-MotionTrackingDeploy`，其中默认推理主线仍是 166D VelCmdHistory 双输入 ONNX。

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

- `teleopit/interfaces.py`
  - 定义稳定协议：`InputProvider`、`Retargeter`、`Controller`、`Robot`、`ObservationBuilder`、`Recorder`。
- `teleopit/runtime/`
  - 负责配置解析、路径归一、组件装配、CLI 公共校验。
- `teleopit/pipeline.py`
  - offline / online sim 的轻量 façade。
- `teleopit/sim2real/controller.py`
  - 真机状态机与控制行为。
- `teleopit/controllers/observation.py`
  - 只暴露 `VelCmdObservationBuilder`。
- `teleopit/controllers/rl_policy.py`
  - 只接受 166D 双输入 VelCmdHistory ONNX。
- `train_mimic/app.py`
  - train/play/benchmark 共用 task 装配与校验。
- `train_mimic/tasks/tracking/config/`
  - 只保留一个公开 task 注册面。
- `train_mimic/data/dataset_builder.py`
  - 数据集构建唯一正式入口。

## 当前公共面

- 稳定运行模式：offline sim2sim、UDP online sim2sim、Pico4 sim2sim、G1 sim2real
- 稳定训练入口：`train.py`、`play.py`、`benchmark.py`、`save_onnx.py`
- 稳定数据入口：`build_dataset.py`

## 约束

- `controller.policy_path` 必须显式提供且文件必须存在。
- 离线 BVH 运行必须显式提供 `input.bvh_file`。
- `viewers` 是唯一 viewer 配置入口。
- 观测定义与 ONNX 输入维度不匹配时启动即报错。
- sim2real 也只支持 166D VelCmdHistory 双输入 ONNX。
