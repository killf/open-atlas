# Replica 与精简 Atlas 实现方案

历史 RGBD 基线记录；当前版本已改为仅用 RGB 预测下一帧，见 [模型说明](rgb-next-frame.md)。
本轮实现下载、轨迹渲染、训练和推理的最小闭环。使用 Python 标准库下载 Replica 官方 v1.0 分卷，以流式解压仅保存选定场景；不把 SLAM 已渲染视频当成原始三维资产。HTTP 中断按 Range 续传；重新启动时核验已完成文件，gzip 仍需从起点解码。

渲染使用已有 Blender，输出 RGB PNG、相机 Z 深度 NPY、OpenCV 相机内外参和 MP4。Replica 使用专用 PTex 纹理；本轮轻量 Blender 路径明确使用原始网格顶点色近似，并以体素聚类简化几何，不宣称复现官方 PTex 外观。体素大小可设为 0 保留原网格。相机采用小范围平移、旋转及复合轨迹，允许显式指定位置，保存坐标归一化变换。

模型以参考 RGB 和相机参数为条件，联合预测 RGB 与深度，采用 rectified flow 速度回归。参考 RGB 与 Plücker 射线共 9 通道，噪声目标 RGBD 与射线共 10 通道，分别编码后共享小型 Transformer。每次生成一帧，将已生成 RGB 加入上下文，形成自回归序列；训练和推理均不读取上下文真值深度。合成深度只用作目标监督和评测标注，推理时所有真值深度及其元数据字段均可省略。

Atlas 官方明确支持 RGB 图像或视频输入并预测几何，但没有公开其完整训练数据配方，不能据此断言官方训练只用 RGB。当前方案是有合成深度监督的 RGB 条件小模型，也仍需要已知相机内外参。参考 Atlas 官方技术文章、DiT、Rectified Flow、CameraCtrl 和 Diffusion Forcing；不声称实现未公开的完整 Atlas、文本分支、潜空间 VAE 或三维高斯预测。

每帧文件及相机参数由 transforms.json 对齐。上下文仅来自目标帧之前；训练、验证、测试以 scene_id 隔离。深度编码、相机坐标、下载恢复和安全解压采用针对性测试；CPU 上以低分辨率完成真实数据训练与采样的冒烟验证，不将其作为质量复现结论。

## 文件与接口

- `scripts/download_replica.py`：官方分卷流、筛选解压、SHA256 清单、probe 和 verify-only。
- `scripts/render_replica.py`：Blender 导入顶点色网格、相机轨迹、RGB/深度/视频导出。
- `open_atlas/data.py`：场景隔离、帧与相机读取、深度编码和相机射线。
- `open_atlas/geometry.py`、`open_atlas/mesh.py`：统一相机约定、轨迹与顶点色网格准备。
- `open_atlas/model.py`：紧凑 RGBD Transformer、流匹配损失、Euler 采样。
- `open_atlas/train.py`、`open_atlas/sample.py`：训练、检查点、逐帧生成和点云导出。
- `tests/`：本地 HTTP 与分卷归档测试、几何约定、数据泄漏和模型训练测试。

## 验证与交付

先验证缺失实现的测试失败，再实现通过。实际检查官方源 Range 响应、下载一个 Replica 场景网格并复核哈希；渲染若干真实帧，执行训练、推理并检查结果文件。记录实际执行的参数、结果及限制于 `docs/validation.md`。用户已授权本轮实现及验证，沿用当前 `work` 分支并保留已有未提交文档。
