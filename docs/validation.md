# Replica 与精简 Atlas 验证记录

验证日期：2026 年 10 月 20 日。环境为 Python 3.12、PyTorch 2.14.1 CPU、Blender 4.3.2 和 FFmpeg，无 CUDA GPU。结构化数据见 [validation-results.json](validation-results.json)。

## 官方场景下载

执行：

```bash
python scripts/download_replica.py --probe
python scripts/download_replica.py --scenes apartment_0 --geometry-only --max-gb 10
python scripts/download_replica.py --scenes apartment_0 --geometry-only --verify-only
```

官方 GitHub Release 返回 HTTP 206，范围探测为 `bytes 0-65535/2000000000`，前缀为 gzip。实际流式读取 5,360,547,840 个压缩字节，提取 `apartment_0/mesh.ply`，没有把整个分卷归档保存到磁盘。

| 项目 | 实际结果 |
| --- | --- |
| 网格大小 | 200,778,055 字节，约 191.5 MiB |
| SHA256 | `274677d9b7caa413230363b68f4aa472e5cc1afe87e422a489fa1b7d844c6182` |
| 原始顶点 | 4,564,613 |
| 原始四边形 | 4,560,777 |
| 文件验证 | 独立复算 SHA256 与脚本 verify-only 均通过 |

这是官方原始网格的几何子集，不包含 PTex 文件。SHA256 用于本地一致性检查，官方没有可对照的可信文件校验清单。完整场景下载路径另由本地分卷和归档测试覆盖，本次没有下载整套 PTex 或其他 Replica 场景。

## 三维相机轨迹渲染

使用以下参数，分别运行 `truck pan compound` 和 `orbit dolly crane` 两组：

```bash
python scripts/render_replica.py \
  --scene data/raw/replica/apartment_0 \
  --output data/rendered/replica_smoke \
  --position 0 0 1 --target 0 -1 1 \
  --trajectories truck pan compound --frames 8 --size 64
```

六类轨迹各 8 帧，共 48 帧，64 × 64、12 fps。逐帧检查 RGB/深度形状、深度有限值、`c2w @ w2c` 为单位矩阵，以及 MP4 帧数、分辨率与帧率。有效深度覆盖率最低为 97.88%，其余无效像素保留为 0。抽查了相机轨迹图像，能够看到房间视差与转向。

颜色采用原始顶点色并使用自发光材质，几何采用默认 0.025 体素聚类近似。另用宽视角正对平面的 Blender 集成测试验证：相机 Z 为 2 时，整张深度图均为 2；顶点色经颜色空间转换后保持预期 RGB。这一测试覆盖深度、图像方向、相机转换与材质路径，避免把径向距离误作 Z 深度。

## 训练与自回归采样

首先执行了 120 步训练与 8 步积分采样，确认检查点和视频能生成。随后在六类轨迹的 42 个可用目标帧上执行小模型训练：

```bash
python -m open_atlas.train --data data/rendered/replica_smoke \
  --output runs/replica-mini --steps 1200 \
  --image-size 32 --dim 64 --layers 2 --batch-size 4 --device cpu

python -m open_atlas.sample --checkpoint runs/replica-mini/last.pt \
  --trajectory data/rendered/replica_smoke/apartment_0_truck_000/transforms.json \
  --output outputs/replica-mini --steps 20 --device cpu
```

小模型有 127,104 个参数。训练前 20 步平均流损失为 1.28096，最后 20 步为 0.31691。以 1 帧真实 RGBD 为参考，自回归生成后续 7 帧，输出 RGB 为 `(7, 32, 32, 3)`、深度为 `(7, 32, 32)`，所有深度为有限数值，并生成 MP4、点云及相机元数据。该小模型预览仍有明显噪声，损失下降不能作为生成质量达标的证据。

随后使用默认骨干配置执行了 3000 步单场景拟合：

```bash
python -m open_atlas.train --data data/rendered/replica_smoke \
  --output runs/replica-fit --steps 3000 --image-size 32 \
  --dim 128 --layers 4 --batch-size 4 --device cpu

python -m open_atlas.sample --checkpoint runs/replica-fit/last.pt \
  --trajectory data/rendered/replica_smoke/apartment_0_truck_000/transforms.json \
  --output outputs/replica-fit --steps 40 --device cpu
```

该配置有 863,680 个参数，前 20 步平均流损失为 1.17760，最后 20 步为 0.10990。40 步积分生成 7 帧 32 × 32 视频，12 fps，点云包含 6486 个有效点。预览能看到房间的大致结构，但仍有明显噪声与模糊，尚不构成高质量新视角合成或三维一致性的验证。

## 自动化与独立审查

测试覆盖官方源之外的本地可控输入：HTTP 连接中断后精确按偏移恢复、服务器忽略 Range 时拒绝续传、gzip 跨分卷、损坏文件修复、路径穿越和符号链接拒绝、不完整 gzip 和历史清单不能误判场景完成。

数据与模型测试覆盖相机投影和 Plücker 射线、内参缩放、深度编解码、场景 split 隔离、仅使用历史帧、单帧上下文使用最新生成帧、梯度与短训练、检查点恢复和视频点云导出。端到端测试在第一次采样后删除所有目标 RGBD 真值，再次采样得到相同结果；推理不依赖目标真值。

完整测试实际运行结果为 **22 passed**，包含 Blender 渲染集成测试。执行了独立代码审查，重要问题均补充失败用例后修复。项目安装、依赖一致性检查、Python 编译检查及 Git 空白检查通过。

## 结果适用范围

本次验证覆盖下载与实现闭环，以及单场景拟合，不包含未见场景评测、Atlas 官方指标对比或长期三维一致性质量结论。当前生成器需要参考 RGBD，未实现仅 RGB 输入时的深度估计、文本条件、动态物理、VAE 或三维高斯生成。导出的点云是预测深度的反投影，不能视作已完成高质量三维重建。
