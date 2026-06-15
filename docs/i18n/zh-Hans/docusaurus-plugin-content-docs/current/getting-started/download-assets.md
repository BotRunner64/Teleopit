---
sidebar_position: 2
---

# 下载资源

机器人模型、数据集和检查点托管在 ModelScope 上，使用前需要先下载。

## 一键下载

下载全部资源（模型、数据、GMR 重定向资源）：

```bash
pip install modelscope
python scripts/setup/download_assets.py
```

## 按需下载

只下载推理所需的资源：

```bash
python scripts/setup/download_assets.py --only robots gmr ckpt bvh
```

## 资源清单

| 资源 | 大小 | 用途 |
|------|------|------|
| `track.onnx` | 4 MB | ONNX 推理模型 |
| `track.pt` | 27 MB | PyTorch 检查点（用于恢复训练） |
| `data/datasets/seed/shard_*.h5` | ~26 GB | 训练数据集 |
| `data/sample_bvh/*.bvh` | 5 MB | 示例动捕文件 |
| `assets/robots/unitree_g1/` | ~52 MB | 训练、sim2sim、重定向和 FK 校验共用的 G1 canonical XML 与 mesh |
| `teleopit/retargeting/gmr/assets/` | ~1.2 GB | GMR 重定向资源、IK 配置和非 canonical 机器人描述 |

## 资源分组

| 分组 | ModelScope 仓库 | 包含内容 |
|------|-----------------|----------|
| `ckpt` | `BingqianWu/Teleopit-models` | `track.onnx`、`track.pt` |
| `robots` | `BingqianWu/Teleopit-models` | Canonical 机器人 XML/mesh |
| `gmr` | `BingqianWu/Teleopit-models` | GMR 重定向资源 |
| `bvh` | `BingqianWu/Teleopit-models` | 示例 BVH 动捕文件 |
| `data` | `BingqianWu/Teleopit-datasets` | 训练 / 验证数据分片 |

资源管理的更多细节（上传、版本控制等）请参阅 [资源管理](../reference/assets)。
