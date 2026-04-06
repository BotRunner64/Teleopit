# 资源管理

机器人模型、数据集、checkpoint 和演示媒体不进 Git 历史，统一走外部下载。

## 不入库的内容

- `teleopit/retargeting/gmr/assets/`：机器人 mesh、URDF/MJCF 等
- `train_mimic/assets/`：已不再跟踪；训练侧 FK 工具直接复用 `teleopit/retargeting/gmr/assets/unitree_g1/g1_mjlab.xml`
- `data/`、checkpoint、cache 等生成产物
- 演示媒体（`assets/demo.gif`、`assets/demo.mp4`）

## ModelScope 仓库

| 仓库 | 类型 | 内容 |
|------|------|------|
| `BingqianWu/Teleopit-models` | model | checkpoint、GMR retargeting 资源、示例 BVH |
| `BingqianWu/Teleopit-datasets` | dataset | 训练/验证数据集 |

资源组与仓库的对应关系：

| 组 | 仓库 | 远端路径 |
|----|------|---------|
| `ckpt` | Teleopit-models | `checkpoints/track.onnx`, `checkpoints/track.pt` |
| `gmr` | Teleopit-models | `archives/gmr_assets.tar.gz` |
| `bvh` | Teleopit-models | `archives/sample_bvh.tar.gz` |
| `data` | Teleopit-datasets | `data/train/`, `data/val/` |

## 下载方式

用项目自带的下载脚本，不要手动提交资源文件：

```bash
python scripts/setup/download_assets.py
python scripts/setup/download_assets.py --only gmr
python scripts/setup/download_assets.py --only ckpt bvh
```

下载脚本会按 repo 类型分别调用 ModelScope API，并将文件放置到对应本地路径：

- `checkpoints/track.onnx` → `track.onnx`
- `checkpoints/track.pt` → `track.pt`
- `archives/gmr_assets.tar.gz` → `teleopit/retargeting/gmr/assets/`（解压）
- `archives/sample_bvh.tar.gz` → `data/sample_bvh/`（解压）
- `data/train/` → `data/datasets/seed/train/`
- `data/val/` → `data/datasets/seed/val/`

## 上传 ModelScope

### 步骤一：准备上传目录

```bash
python scripts/setup/prepare_modelscope_assets.py --only ckpt gmr bvh --clean
python scripts/setup/prepare_modelscope_assets.py --only data
```

产物在 `data/modelscope_upload/`。

### 步骤二：上传到对应仓库

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

### 步骤三：打版本 tag

ModelScope 仅模型仓库支持 tag，数据集仓库不支持。

```bash
python - <<'EOF'
from modelscope.hub.api import HubApi
api = HubApi()
url = api.create_model_tag("BingqianWu/Teleopit-models", "v0.2.0")
print(url)
EOF
```

tag 与代码仓库的 Git tag 保持一致，方便追溯每个版本对应的模型。

## 提交前检查

推送前跑一下仓库卫生检查：

```bash
python scripts/check_large_tracked_files.py
```

会拦截大二进制文件，并检查已跟踪文件的体积上限。

## 历史瘦身

如需缩小 GitHub clone 体积，在干净的 mirror clone 里重写历史：

```bash
git clone --mirror https://github.com/BotRunner64/Teleopit.git Teleopit.git
cd Teleopit.git
git filter-repo \
  --force \
  --path teleopit/retargeting/gmr/assets \
  --path train_mimic/assets \
  --path assets/demo.gif \
  --path assets/demo.mp4 \
  --invert-paths
git push --force --mirror
```

force-push 后，协作者需要重新 clone，不要在旧仓库上修补。
