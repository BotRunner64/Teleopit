---
sidebar_position: 2
---

# 资源管理

机器人模型、数据集、checkpoint 和演示媒体不进 Git 历史，统一走外部下载。

## 不入库的内容

- `teleopit/retargeting/gmr/assets/` — 机器人 mesh、URDF/MJCF 等
- `data/`、checkpoint、缓存等生成产物
- 演示媒体（`assets/demo.gif`、`assets/demo.mp4`）

## 远程仓库

### ModelScope（默认下载源）

| 仓库 | 类型 | 内容 |
|------|------|------|
| `BingqianWu/Teleopit-models` | model | checkpoint、GMR retargeting 资源、示例 BVH |
| `BingqianWu/Teleopit-datasets` | dataset | 训练/验证数据集 |

### HuggingFace（备选）

| 仓库 | 类型 | 内容 |
|------|------|------|
| `12e21/Teleopit-models` | model | checkpoint、GMR retargeting 资源、示例 BVH |
| `12e21/Teleopit-datasets` | dataset | 训练/验证数据集 |

### 资源组与仓库的对应关系

| 组 | 仓库 | 远端路径 |
|----|------|---------|
| `ckpt` | Teleopit-models | `checkpoints/track.onnx`、`checkpoints/track.pt` |
| `gmr` | Teleopit-models | `archives/gmr_assets.tar.gz` |
| `bvh` | Teleopit-models | `archives/sample_bvh.tar.gz` |
| `data` | Teleopit-datasets | `data/` |

## 下载

使用项目自带的下载脚本（默认从 ModelScope 下载）：

```bash
# 下载全部
python scripts/setup/download_assets.py

# 只下载推理必需的资源
python scripts/setup/download_assets.py --only gmr ckpt bvh

# 只下载训练数据
python scripts/setup/download_assets.py --only data

# 从 HuggingFace 下载
python scripts/setup/download_assets.py --source huggingface
```

下载后各资源的本地落点：

| 远端路径 | 本地路径 |
|---------|---------|
| `checkpoints/track.onnx` | `track.onnx` |
| `checkpoints/track.pt` | `track.pt` |
| `archives/gmr_assets.tar.gz` | `teleopit/retargeting/gmr/assets/`（自动解压） |
| `archives/sample_bvh.tar.gz` | `data/sample_bvh/`（自动解压） |
| `data/` | `data/datasets/seed/` |

## 上传到 ModelScope

### 第一步：准备上传目录

```bash
python scripts/setup/prepare_modelscope_assets.py --only ckpt gmr bvh --clean
python scripts/setup/prepare_modelscope_assets.py --only data
```

产物在 `data/modelscope_upload/`。

### 第二步：上传到对应仓库

```bash
# 模型仓库
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/checkpoints checkpoints
modelscope upload --repo-type model BingqianWu/Teleopit-models \
    data/modelscope_upload/archives archives

# 数据集仓库
modelscope upload --repo-type dataset BingqianWu/Teleopit-datasets \
    data/modelscope_upload/data data
```

### 第三步：打版本 tag

ModelScope 仅模型仓库支持 tag，数据集仓库不支持。

```bash
python - <<'EOF'
from modelscope.hub.api import HubApi
api = HubApi()
url = api.create_model_tag("BingqianWu/Teleopit-models", "vX.Y.Z")
print(url)
EOF
```

tag 与代码仓库的 Git tag 保持一致，方便追溯每个版本对应的模型。

## 上传到 HuggingFace

### 第一步：准备 staging 目录

```bash
# 准备全部（--clean 会清空旧的 staging 目录）
python scripts/setup/upload_hf_assets.py --dry-run --clean

# 只准备指定组
python scripts/setup/upload_hf_assets.py --only ckpt gmr bvh --dry-run
python scripts/setup/upload_hf_assets.py --only data --dry-run
```

`--dry-run` 只写 staging，不执行上传，可用于检查文件完整性。

### 第二步：执行上传

```bash
python scripts/setup/upload_hf_assets.py --only ckpt gmr bvh --clean
python scripts/setup/upload_hf_assets.py --only data --clean
```

:::warning
每次运行前建议加 `--clean`，否则 staging 目录可能残留上次遗留的文件，导致 `--only` 语义失效（旧资源被误带入上传）。
:::

### 第三步：打版本 tag

HuggingFace 模型仓库支持 tag（数据集仓库不支持）：

```bash
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.create_tag("12e21/Teleopit-models", tag="vX.Y.Z", repo_type="model")
EOF
```

tag 与代码仓库的 Git tag 保持一致，方便追溯每个版本对应的模型。

## 提交前检查

推送代码前运行：

```bash
python scripts/dev/check_large_tracked_files.py
```

会拦截大二进制文件并检查已跟踪文件的体积上限。
