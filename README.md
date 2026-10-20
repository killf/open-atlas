# open-atlas

基于 World Labs Atlas 公开机制的小规模 **RGB 下一帧预测**实验：Replica 场景 → 单目 RGB 轨迹视频 → 图像潜空间自回归生成。

**深度不参与模型输入、训练目标或损失；本版本也不输出深度。** 输入是历史 RGB 和已知相机参数，目标是下一帧 RGB。自编码器产生的四通道潜变量都是图像特征，并非 RGBD。

Atlas 官方介绍了多模态自回归潜扩散模型，但没有公开完整网络与训练配方。本项目限定其中的 RGB／相机条件任务，不能据此宣称官方只用 RGB 训练。原文对照与实现取舍见 [模型说明](docs/rgb-next-frame.md)。

## 安装与下载

需要 Python 3.11+；合成视频另需 PATH 中的 Blender 4.3 与 FFmpeg。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
python scripts/download_replica.py --probe
python scripts/download_replica.py --scenes apartment_0 apartment_1 apartment_2 --geometry-only
python scripts/download_replica.py --scenes apartment_0 apartment_1 apartment_2 --geometry-only --verify-only
python scripts/download_codec.py
python scripts/download_codec.py --verify-only
```

Replica 从官方 17 个分卷流式提取指定网格，保存许可和 SHA256；HTTP 中断通过 Range 恢复，重新运行 gzip 仍须从头解码。小子集节省本地存储，不能跳过前面的压缩字节。遵守 [Replica Research Terms](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE)，限非商业研究或教育用途。

图像编码使用冻结的 [TAESD](https://github.com/madebyollin/taesd) 轻量自编码器。下载脚本固定作者仓库版本并校验 SHA256，约 9.4 MiB；许可见 [TAESD MIT License](docs/asset-licenses/TAESD.txt)。这是替代 Atlas 未公开编码器的工程选择。

## 合成 RGB 视频

```bash
python scripts/render_replica.py \
  --scene data/raw/replica/apartment_0 --output data/rendered/replica \
  --position 0 0 0 --target 0 -1 0 --frames 24 --size 64 --split train
```

默认生成环绕、推拉、横移、升降、摇摄和复合轨迹，保存无损 RGB、相机参数与预览视频。相机采用 OpenCV 右／下／前约定，像素中心为 `(x+0.5,y+0.5)`。机位使用原始场景坐标，需按房间设置并预览检查；当前不自动规划避障。

渲染使用网格顶点色近似，不是 Replica 原始 PTex 纹理。网格最长边归一化为 4，变换保存在元数据中。`--depth` 可额外导出独立分析标注，但训练和评测不会读取它。

```text
apartment_0_truck_000/
  rgb/000000.png
  transforms.json     # K、c2w、w2c、timestamp、scene_id、split
  preview.mp4
```

按场景划分 train／val／test，同场景轨迹不可跨集合。已合成三场景数据位于 `data/rendered/replica_rgb_v2/`：训练 1152 帧、验证 96 帧、测试 144 帧；本轮 RGB-only 目录硬链接复用旧图像，未包含深度文件或字段；旧目录仍可保留作对照。

## 训练、生成与评测

```bash
python -m open_atlas.train --data data/rendered/replica_rgb_v2 \
  --output runs/rgb-next-frame-v1 --steps 1000 --image-size 64 --validate --seed 42

python -m open_atlas.sample --checkpoint runs/rgb-next-frame-v1/last.pt \
  --trajectory data/rendered/replica_rgb_v2/test/apartment_2_truck_200/transforms.json \
  --output outputs/rgb-next-frame-sample --steps 20

python -m open_atlas.evaluate --checkpoint runs/rgb-next-frame-v1/last.pt \
  --data data/rendered/replica_rgb_v2 --output outputs/rgb-next-frame-eval \
  --split test --steps 20 --metric-size 32
```

64×64 RGB 被编码为 4×8×8 图像潜变量。编码器冻结，训练前仅编码一次；4 层、128 维 DiT 使用 adaLN-Zero 条件调制，以 rectified flow 预测下一帧潜变量的速度。相机转换到第一参考帧坐标系，视频时间间隔与扩散时间分别编码。

训练上下文只来自目标之前的帧；每次仅监督一个下一帧目标。推理读取开头的参考 RGB，后续将生成的图像潜变量加入历史，不读取目标图像。输出为 RGB、MP4、相机元数据与 `rgb.npz`。旧 RGBD 检查点不兼容；当前仍需已知相机内外参，未实现从无位姿 MP4 估计相机。

评测采用单张真实参考图像的自由自回归滚动，报告 RGB PSNR／MAE 和重复首帧基线；默认统一在 32×32 评分，便于与历史实验比较。单场景小模型不能据此宣称 Atlas 级生成质量或泛化能力。

## 验证与文档

```bash
python -m pytest -q
```

测试覆盖无深度训练、下一帧目标与历史隔离、世界坐标变换不变性、冻结图像编码器、流匹配、无未来真值依赖的滚动生成，以及真实 Blender 渲染。

- [模型与原文对照、1000 步结果](docs/rgb-next-frame.md)
- [相关论文](docs/atlas-research.md)与[已提交的 PDF](docs/papers/README.md)
- [旧版本代码审查](docs/atlas-code-audit.md)：对应旧 RGBD 输出基线，保留作对照
- [历史数据合成记录](docs/experiments/replica-mini-v1.md)

项目名 `open-atlas`，Python 包名 `open_atlas`；注释说明算法、形状和坐标约定。原始数据、潜变量缓存、权重和生成视频不纳入 Git。
