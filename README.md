# open-atlas

基于 World Labs Atlas 公开技术思路的精简世界模型实验：Replica 三维场景 → 相机轨迹 RGBD 数据 → 相机条件生成 → 自回归视频与点云。

这是可运行的小规模实现，尚未复现完整 Atlas 的能力或论文指标。官方未公开完整训练实现；本项目从零训练像素空间 RGBD Transformer，参考 [Atlas 官方文章](https://www.worldlabs.ai/blog/atlas)、[DiT](https://arxiv.org/abs/2212.09748)、[Rectified Flow](https://arxiv.org/abs/2209.03003) 和 [CameraCtrl](https://arxiv.org/abs/2404.02101)。

## 安装

需要 Python 3.11+、Blender 4.3 和 FFmpeg，后两者须在 PATH 中。仅下载数据时不需要 PyTorch 或 Blender。

```bash
python -m venv .venv
source .venv/bin/activate
# CPU 环境；GPU 环境可按 PyTorch 官方说明安装对应 CUDA 版本。
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
```

## 下载 Replica 并验证

采用 [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset) 原始三维场景，不依赖已有的 SLAM 渲染轨迹。默认选取首个场景 `apartment_0`。

```bash
python scripts/download_replica.py --probe
python scripts/download_replica.py --scenes apartment_0 --geometry-only
python scripts/download_replica.py --scenes apartment_0 --geometry-only --verify-only
```

脚本流式拼接官方 v1.0 的 17 个分卷，只保存选定场景。`--geometry-only` 仅保留带顶点色的 `mesh.ply`；去掉该选项可保留场景其余文件，包括 PTex。默认写入 `data/raw/replica/`，包含原始许可和逐文件 SHA256 清单。校验失败时 `--verify-only` 返回非零退出码。

网络中断通过 HTTP Range 续传；重新启动可复核、跳过已完成文件，但 gzip 必须从头重新读取。选小子集节省的是本地存储，不能跳过此前压缩数据：本次提取约 191.5 MiB 的网格实际读取约 5 GiB。可用 `--max-gb 8` 限制单次读取量。SHA256 是本地完整性记录，官方未提供可信哈希清单。

使用时遵守 [Replica Dataset Research Terms](https://github.com/facebookresearch/Replica-Dataset/blob/main/LICENSE)，限非商业或非营利性质的研究、教育用途。

## 渲染相机轨迹

```bash
python scripts/render_replica.py \
  --scene data/raw/replica/apartment_0 \
  --output data/rendered/replica \
  --position 0 0 1 --target 0 -1 1 \
  --frames 48 --size 64 --split train
```

默认生成环绕、推拉、横移、升降、摇摄和复合轨迹。`--position`、`--target` 使用原始 Replica 坐标；`--amplitude` 默认 0.08，平移轨迹使用归一化场景单位，环绕和摇摄使用弧度。相机位置须按场景设置，当前不做自动碰撞规划。

Replica 的高分辨率纹理采用专用 PTex；此处 Blender 路径使用**网格顶点色近似**，不是官方 PTex 渲染器。默认以体素聚类简化几何，`--voxel-size 0` 可保留完整网格。世界坐标被平移、缩放至最长边 4 个单位，转换矩阵随数据保存；深度和点云采用该单位。

每段数据包含：

```text
apartment_0_truck_000/
  rgb/000000.png       # 无损 RGB；训练读取这些图像
  depth/000000.npy     # float32 相机 Z 深度，0 表示无效
  transforms.json     # K、c2w、w2c、时间、场景、split 与坐标约定
  preview.mp4         # 仅用于预览
```

相机采用 OpenCV 约定：右、下、前，像素中心为 `(x+0.5, y+0.5)`。按场景划分 train / val / test，同一场景的所有轨迹必须属于同一 split；加载器会拒绝跨集合场景泄漏。

## 训练与生成

```bash
python -m open_atlas.train --data data/rendered/replica \
  --output runs/open-atlas --image-size 32 --steps 3000

python -m open_atlas.sample --checkpoint runs/open-atlas/last.pt \
  --trajectory data/rendered/replica/apartment_0_truck_000/transforms.json \
  --output outputs/sample --steps 20
```

默认骨干为 4 层、128 维的 Transformer。RGB 与对数编码深度联合构成 4 通道，相机 Plücker 射线提供空间条件。训练使用 `x_t = (1-t) * noise + t * target`，回归速度 `target - noise`；采样使用 Euler 积分。过去帧不能关注目标 token，训练上下文只取目标之前的帧。`--context-frames 1` 时使用最近一帧；更多上下文在历史中均匀选取。

推理读取开头的参考 RGBD 和后续相机位姿，此后将**生成的 RGBD** 加入历史，不读取目标 RGBD 真值。`transforms.json` 可替换目标相机轨迹；仅参考帧需要可读取的 RGB 和深度文件。输出生成帧、深度、MP4、相机参数和深度反投影点云 `points.ply`。输出目录须为新目录，避免覆盖已有结果。

准备了独立场景的 val 数据后，可给训练命令增加 `--validate` 记录验证流损失。单场景短训练只能验证流程及拟合，不能据此评价未见场景泛化。当前不含文本条件、VAE、预训练大模型、物理动态或三维高斯生成；点云由预测深度反投影得到。

## 验证

```bash
python -m pytest -q
```

测试覆盖分卷拼接、HTTP 中断恢复、安全解压、截断归档与损坏检测、内参缩放、深度编码、场景隔离、单帧与多帧自回归、训练与检查点恢复。安装 Blender 和 FFmpeg 时还会实际渲染平面，核对相机 Z 深度和顶点色。

真实下载、渲染、CPU 训练和采样结果见 [验证记录](docs/validation.md)。数据、权重和生成结果不纳入 Git。

## 代码约定

项目名统一为 `open-atlas`，Python 导入包为 `open_atlas`。代码保持直接、简洁；注释重点解释算法、张量形状、坐标约定和边界处理，依赖与抽象仅按实际需要引入。

## 文档

- [Atlas 相关论文与三维训练数据方案](docs/atlas-research.md)：官方技术资料、相关论文、mini 三维资产选型与相机轨迹视频规范。
- [本地论文 PDF](docs/papers/README.md)：已下载的 CAT3D、BlenderProc2 原文及来源、SHA256 清单，随代码提交。
- [实现方案](docs/implementation.md)：模块边界、论文对应关系与复现范围。
- [首批三维资产清单](docs/asset-manifest.json)：下载地址、来源版本、文件大小、SHA256 和署名文件。

早期下载的 Khronos Avocado、ToyCar、Fox 保存在 `data/raw/khronos/`，仅作为辅助调试资产；本轮训练和验证使用 Replica。
