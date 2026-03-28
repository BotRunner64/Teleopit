# 资源管理

机器人模型、数据集、checkpoint 和演示媒体不进 Git 历史，统一走外部下载。

## 不入库的内容

- `teleopit/retargeting/gmr/assets/`：机器人 mesh、URDF/MJCF 等
- `train_mimic/assets/`：已不再跟踪；训练侧 FK 工具直接复用 `teleopit/retargeting/gmr/assets/unitree_g1/g1_mjlab.xml`
- `data/`、checkpoint、cache 等生成产物
- 演示媒体（`assets/demo.gif`、`assets/demo.mp4`）

## 下载方式

用项目自带的下载脚本，不要手动提交资源文件：

```bash
python scripts/download_assets.py
python scripts/download_assets.py --only gmr
python scripts/download_assets.py --only ckpt bvh
```

`gmr` 是推理和训练 FK 工具都需要的机器人资源组。

ModelScope 远端把小文件多的目录打成压缩包存储：

- 仓库：`BingqianWu/Teleopit-assets`
- 地址：https://www.modelscope.cn/models/BingqianWu/Teleopit-assets
- `archives/gmr_assets.tar.gz`
- `archives/sample_bvh.tar.gz`

下载脚本会自动解压到本地对应目录：

- `archives/gmr_assets.tar.gz` -> `teleopit/retargeting/gmr/assets/`
- `archives/sample_bvh.tar.gz` -> `data/sample_bvh/`
- `checkpoints/track.onnx` -> `track.onnx`
- `checkpoints/track.pt` -> `track.pt`
- `data/train/` -> `data/datasets/seed/train/`
- `data/val/` -> `data/datasets/seed/val/`

## 上传 ModelScope

更新 ModelScope 仓库前，先在本地生成上传目录：

```bash
python scripts/prepare_modelscope_assets.py --only gmr bvh --clean
python scripts/prepare_modelscope_assets.py --only ckpt data
```

产物在 `data/modelscope_upload/`，上传时保持相对路径：

```bash
modelscope upload --repo-type model BingqianWu/Teleopit-assets \
  data/modelscope_upload/archives archives
modelscope upload --repo-type model BingqianWu/Teleopit-assets \
  data/modelscope_upload/checkpoints checkpoints
modelscope upload --repo-type model BingqianWu/Teleopit-assets \
  data/modelscope_upload/data data
```

- `gmr` 和 `bvh` 在 ModelScope 上保持 `.tar.gz` 压缩包形式，避免大量碎文件
- checkpoint 和 dataset shard 直接按文件/目录上传

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

## 提交前检查

推送前跑一下仓库卫生检查：

```bash
python scripts/check_large_tracked_files.py
```

会拦截大二进制文件，并检查已跟踪文件的体积上限。
