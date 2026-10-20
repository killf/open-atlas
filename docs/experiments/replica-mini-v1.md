# Replica mini v1：数据合成与首轮训练

本轮采用三个官方 Replica 场景，按场景划分数据；相机机位在原始场景坐标中指定。生成数据位于 `data/rendered/replica_mini_v1/`，不与之前的 `replica_smoke` 混用。

## 数据配置

| 划分 | 场景 | 视频数 | 每段帧数 | RGB 帧数（附深度标注） | 可用目标样本 |
| --- | --- | ---: | ---: | ---: | ---: |
| 训练 | apartment_0 | 48 | 24 | 1152 | 1104 |
| 验证 | apartment_1 | 6 | 16 | 96 | 90 |
| 测试 | apartment_2 | 6 | 24 | 144 | 138 |

每段第一帧只提供上下文，其余帧可作为目标。六类轨迹为环绕、推拉、横移、升降、摇摄、复合运动；训练场景采用四个机位，每个机位两个随机种子。种子改变轨迹的起始相位，不代表新增场景。验证、测试各采用一个机位。

统一渲染为 64×64、12 fps，FOV 60°，轨迹幅度 0.08，体素大小 0.025。颜色为网格顶点色近似。无效深度保存为 0，并在深度损失中屏蔽；深度单位是归一化场景单位。机位经预览检查，避开了靠近天花板或正对近处墙面的视角；测试机位选择只依据场景完整性，与模型输出无关。

完整参数见 [replica-mini-v1.json](replica-mini-v1.json)。逐帧 SHA256、相机与视频校验结果保存在数据目录的 `dataset-manifest.json`。

## 生成命令

在项目根目录执行；输出目录必须是新目录。

```bash
python scripts/download_replica.py --scenes apartment_0 apartment_1 apartment_2 --geometry-only
python scripts/download_replica.py --scenes apartment_0 apartment_1 apartment_2 --geometry-only --verify-only
python scripts/render_replica.py --scene data/raw/replica/apartment_0 --output data/rendered/replica_mini_v1/train_view_0 --position 0 0 0 --target 0 -1 0 --frames 24 --size 64 --seed 0 --seeds 2 --split train
python scripts/render_replica.py --scene data/raw/replica/apartment_0 --output data/rendered/replica_mini_v1/train_view_1 --position -0.35 -0.1 0.1 --target -0.1 -1.3 -0.1 --frames 24 --size 64 --seed 10 --seeds 2 --split train
python scripts/render_replica.py --scene data/raw/replica/apartment_0 --output data/rendered/replica_mini_v1/train_view_2 --position 0.35 -0.35 -0.1 --target -0.2 -1.4 -0.2 --frames 24 --size 64 --seed 20 --seeds 2 --split train
python scripts/render_replica.py --scene data/raw/replica/apartment_0 --output data/rendered/replica_mini_v1/train_view_3 --position 0 -0.7 0.25 --target -0.35 -1.8 0 --frames 24 --size 64 --seed 30 --seeds 2 --split train
python scripts/render_replica.py --scene data/raw/replica/apartment_1 --output data/rendered/replica_mini_v1/val --position 0 0 0 --target 0 1 0 --frames 16 --size 64 --seed 100 --seeds 1 --split val
python scripts/render_replica.py --scene data/raw/replica/apartment_2 --output data/rendered/replica_mini_v1/test --position 5 0.5 -0.1 --target 2 0.5 -0.1 --frames 24 --size 64 --seed 200 --seeds 1 --split test
```

## 首轮训练

参考输入只包含 RGB 和相机参数，目标深度用于训练监督；推理时不读取任何真值深度。使用现有精简模型从零训练，本轮启动命令如下：

```bash
python -u -m open_atlas.train \
  --data data/rendered/replica_mini_v1 --output runs/replica-mini-v1-rgb \
  --steps 6000 --image-size 32 --patch-size 4 \
  --dim 128 --layers 4 --heads 4 --batch-size 4 --context-frames 2 \
  --max-depth 10 --lr 0.0003 --seed 42 --device cpu --threads 2 --validate
```

模型含 882,240 个参数。本环境没有 CUDA GPU，训练使用 CPU；模型输入缩放至 32×32。每 20 步记录一次独立验证场景的流损失，验证噪声固定；测试集不参与优化或模型选择。此轮按固定 6000 步训练，不进行验证集最优检查点选择。

运行参数与 PID 写入 `runs/replica-mini-v1-rgb/run.json`，标准输出为 `train.log`，每一步损失为 `metrics.jsonl`，训练完成后保存 `last.pt`。可查看：

```bash
tail -f runs/replica-mini-v1-rgb/train.log
```

训练日志与数据保留在本地，不纳入 Git。该规模用于验证流程与初步拟合，泛化效果需在训练完成后独立评测。

## 本轮检查记录

2026-10-20 已实际生成 60 段 RGB 视频、1392 帧图像及对齐深度监督标注，检查全部帧的尺寸、类型、有限深度、相机逆矩阵及旋转正交性，并用 FFprobe 检查视频分辨率、帧率与帧数。数据加载器确认三个 split 场景互斥，抽查编码张量均为有限值。逐文件哈希保存在本地清单。

训练、验证集的最低有效深度覆盖率均为 100%；测试集最低为 93.99%，平均为 95.49%，其余像素保留无效掩码。

旧的 `runs/replica-mini-v1` 训练要求 RGBD 参考帧，现已停止，日志保留作历史记录。当前模型将上下文改为 3 通道 RGB，生成目标仍为 RGB 和深度，因此需要重新训练；原有渲染数据无需重做。新运行使用 `runs/replica-mini-v1-rgb`，运行参数、原始网格哈希及数据清单哈希见 [结构化记录](replica-mini-v1-report.json)，后续进度以本地日志为准。

RGB 输入修正通过 23 项自动化测试。另在真实数据上完成 40 步冒烟训练，将测试输入精简为一张 RGB、四个相机位姿，移除全部真值深度及目标图像，成功生成后续三帧 RGB 和有限预测深度。这只核对输入接口与执行流程，不评价生成质量。
