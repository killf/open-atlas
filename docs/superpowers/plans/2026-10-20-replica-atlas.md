# Replica 与精简 Atlas Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 可验证地下载 Replica 并运行精简相机条件 RGBD 生成模型。

**Architecture:** 标准库流式下载与场景筛选，Blender 渲染逐帧 RGBD 和相机，PyTorch Transformer 学习 rectified flow，并逐帧采样。

**Tech Stack:** Python 3.11+、NumPy、Pillow、PyTorch 2.4+、pytest、Blender 4.3、FFmpeg。

**Spec:** `docs/implementation.md`

## Global Constraints

- 保持代码简洁，单机运行，不引入训练框架或远程服务。
- 保留原始 Replica 许可与来源，不改变数据集身份。
- 上下文不含目标或未来真值，按 scene_id 划分数据。
- 明确区分模型实现、冒烟验证与完整 Atlas 质量复现。

## Review Focus

- 中断或忽略 Range 的服务器不能造成静默损坏。
- 归档路径和链接不能写出目标目录。
- 不完整的文件或场景不能标记为下载完成。
- 缩放图像后必须同步相机内参；深度和外参约定一致。
- 推理上下文必须使用生成帧；检查点配置必须能还原模型。

## Task 1 下载

- [x] 编写并运行下载测试，证明实现缺失时失败。
- [x] 实现 `MultipartReader`、`extract_scenes`、`verify` 及 CLI。
- [x] 运行测试、官方 probe、实际场景下载与哈希验证。

## Task 2 渲染与数据

- [x] 编写深度、射线、内参缩放及数据划分测试并确认失败。
- [x] 实现 `open_atlas/data.py` 和 Blender 渲染入口。
- [x] 渲染实际场景并检查 RGB、深度、帧数、相机矩阵和视频。

## Task 3 模型与入口

- [x] 编写流积分、条件依赖、因果数据和梯度测试并确认失败。
- [x] 实现 RGBD Transformer、流目标、训练和逐帧采样入口。
- [x] 运行短训练、检查点恢复、采样与点云导出。

## Task 4 验证与说明

- [x] 运行完整测试，核对全部 CLI 示例与 README。
- [x] 独立审查代码，修复重要问题并运行回归验证。
- [x] 保存真实验证结果与可复现命令，不提交数据或模型权重。
