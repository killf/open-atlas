# open-atlas

基于 World Labs Atlas 公开机制的小规模 **RGB 下一帧预测**实验：Replica 场景 → 单目 RGB 轨迹视频 → RGB 像素空间自回归生成。

**深度不参与模型输入、训练目标或损失；本版本也不输出深度。** 输入是历史 RGB 和已知相机参数（包括目标帧相机位姿），目标是下一帧 RGB。当前直接处理三通道 RGB，不使用图像编码器、VAE 或预训练权重。

Atlas 官方介绍了多模态自回归潜扩散模型，但没有公开完整网络与训练配方。本项目限定其中的 RGB／相机条件任务，不能据此宣称官方只用 RGB 训练。像素空间是本项目为简化实验所作的取舍，与官方潜空间方案存在差异；原文对照与当前模型见 [模型说明](docs/rgb-image-overfit.md)。

官方相机条件、普通 RGB 输入的位姿估计前端，以及当前实现取舍，见 [一手资料核对](docs/atlas-source-audit.md)。相机条件有官方依据；当前无位姿估计前端，也未复现 Atlas 的全部模态与潜空间结构。

## 安装与下载

需要 Python 3.11+；合成视频另需 PATH 中的 Blender 4.3 与 FFmpeg。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
python scripts/download_replica.py --probe
python scripts/download_replica.py --scenes apartment_0 apartment_1 apartment_2 --geometry-only
python scripts/download_replica.py --scenes apartment_0 apartment_1 apartment_2 --geometry-only --verify-only
```

Replica 从官方 17 个分卷流式提取指定网格，保存许可和 SHA256；HTTP 中断通过 Range 恢复，重新运行 gzip 仍须从头解码。小子集节省本地存储，不能跳过前面的压缩字节。遵守 [Replica Research Terms](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE)，限非商业研究或教育用途。

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
  --output runs/rgb-image-next-frame-v1 --steps 1000 --image-size 64 --validate --seed 42

python -m open_atlas.sample --checkpoint runs/rgb-image-next-frame-v1/last.pt \
  --trajectory data/rendered/replica_rgb_v2/test/apartment_2_truck_200/transforms.json \
  --output outputs/rgb-image-next-frame-sample --steps 20

python -m open_atlas.evaluate --checkpoint runs/rgb-image-next-frame-v1/last.pt \
  --data data/rendered/replica_rgb_v2 --output outputs/rgb-image-next-frame-eval \
  --split test --steps 20 --metric-size 32
```

64×64 RGB 直接按 4×4 patch 投影为每帧 256 个 token；没有独立编解码器。4 层、128 维 DiT 使用 adaLN-Zero 条件调制，直接预测干净下一帧 RGB，再转换为 rectified-flow 速度计算损失和采样。转换使用 `max(1-t,0.05)` 分母，预测与真值采用相同处理。相机转换到第一参考帧坐标系，视频时间间隔与扩散时间分别编码。

训练上下文只来自目标之前的帧；每次仅监督一个下一帧目标。推理读取开头的参考 RGB，积分末步取干净预测，后续将裁剪到 [0,1] 的生成 RGB 加入历史，不读取目标图像。输出为 RGB、MP4、相机元数据与 `rgb.npz`。当前检查点为格式 4，标记 `parameterization=clean_rgb`；旧速度、RGBD 和潜空间检查点均需重新训练；当前仍需已知相机内外参，未实现从无位姿 MP4 估计相机。

评测采用单张真实参考图像的自由自回归滚动，报告 RGB PSNR／MAE 和重复首帧基线；默认统一在 32×32 评分，便于与历史实验比较。单场景小模型不能据此宣称 Atlas 级生成质量或泛化能力。

历史速度预测像素版完成过 1000 步训练：独立测试场景自由生成 138 帧，在 32×32 评分下 RGB PSNR 为 **11.84 dB**，仍低于复制首帧基线的 17.00 dB。流程已跑通，生成质量仍不足；完整协议和历史对照见 [实验记录](docs/rgb-next-frame.md)。

## 单轨迹过拟合

```bash
python -u -m open_atlas.overfit \
  --trajectory data/rendered/replica_rgb_v2/train_view_0/apartment_0_truck_000/transforms.json \
  --output runs/rgb-image-overfit-v1 --frames 8 --steps 1000 \
  --image-size 64 --sampling-steps 20 --seed 42 --device cpu
```

固定一条训练轨迹的前 8 帧，从头训练 7 个下一帧目标；随后使用真实历史和三个噪声种子做单步评测，输出图像、PSNR／MAE 与复制上一帧基线。此实验只检查训练集记忆能力，协议和结果见 [当前实验](docs/rgb-image-overfit.md)。

本轮 1000 步已完成，但过拟合检查未通过：64×64 单步 PSNR 为 **23.69 dB**，低于复制真实上一帧的 **25.58 dB**。三个噪声种子结果接近；尚不扩大完整数据训练，先定位高噪声端预测误差。命令、哈希与指标见 [结果记录](docs/rgb-image-overfit-results.json)。

## 验证与文档

```bash
python -m pytest -q
```

测试覆盖无深度训练、下一帧目标与历史隔离、世界坐标变换不变性、无编解码器检查点、RGB 裁剪与反馈、流匹配、无未来真值依赖的滚动生成，以及真实 Blender 渲染。

- [当前图像预测模型与单轨迹过拟合](docs/rgb-image-overfit.md)
- [历史速度预测、1000 步结果](docs/rgb-next-frame.md)
- [相关论文](docs/atlas-research.md)与[已提交的 PDF](docs/papers/README.md)
- [旧版本代码审查](docs/atlas-code-audit.md)：对应旧 RGBD 输出基线，保留作对照
- [历史数据合成记录](docs/experiments/replica-mini-v1.md)

项目名 `open-atlas`，Python 包名 `open_atlas`；注释说明算法、形状和坐标约定。原始数据、权重和生成视频不纳入 Git。
