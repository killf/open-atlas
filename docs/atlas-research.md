# Atlas 相关论文与三维训练数据方案

更新日期：2026 年 10 月 20 日。新增核对见 [一手资料与实现边界](atlas-source-audit.md)，包含官方 Beta API、RTFM、相关原论文与固定版本作者代码。

当前任务已限定为 RGB 下一帧像素空间预测，不使用图像编码器或 VAE；深度不参与训练、输入或输出。像素空间是简化实验，与 Atlas 官方潜空间机制不同。下文几何生成与深度监督方案属于此前研究记录，当前已采用干净 RGB 预测参数化，具体实现与单轨迹过拟合见 [模型说明](rgb-image-overfit.md)。

本项目面向 World Labs 的 Atlas 空间世界模型开展小规模开放复现实验。训练数据采用可渲染的三维资产：从多个数据源抽取小子集，设计相机轨迹，渲染连续视频，并同步保存逐帧相机参数和几何标注。第一阶段聚焦相机条件下的新视角生成及三维一致性。

当前已选用 Replica 作为主数据源，完成官方场景网格下载、RGB 轨迹渲染及精简模型实现。下列其他数据源保留为前期调研备选；实际命令与范围以 [README](../README.md) 和 [验证记录](validation.md) 为准。

## Atlas 官方技术资料与复现范围

**World Labs Team，Atlas: A World Model for Spatial Intelligence，2026 年 9 月 1 日。** [官方技术文章](https://www.worldlabs.ai/blog/atlas)。这是一篇公司技术文章，引用时应保留其来源类型。

官方披露的核心是多模态自回归扩散 Transformer：图像、文本、相机位姿和深度组成空间上下文，以 rectified flow 生成连续输出。目标能力包括相机控制生成、空间重建、时空模拟，并可输出点云或 Gaussian splats。

“支持深度模态”并不要求用户提供 RGBD 视频。官方在 Explicit 3D Outputs 部分说明，可从普通视频预测每帧深度并重建三维场景；相机控制与重建评测使用图像和相机位姿。输入所需模态与训练监督应区分：官方没有披露只用 RGB 训练的完整配方。本项目当前只训练 RGB 下一帧，相机参数（包含目标相机）作为条件，深度不参与训练或输出。此前 RGBD 方案仅为历史记录。

截至本次检索，尚未找到能够据以完整重现实验的独立 Atlas 论文和公开训练实现。官方文章没有给出完整的网络配置、训练数据配方、损失权重或检查点。本次已找到官方 Beta API 文档，但服务接口不等于公开训练实现。因此，本项目应标为基于公开思路的缩小版实现；下列论文是可借鉴的方法，不能据此断言 Atlas 采用了它们的具体实现。

官方补充来源：

- [RTFM: A Real-Time Frame Model](https://www.worldlabs.ai/blog/rtfm)，2025-10-16：按目标位姿查询下一帧，使用带位姿历史视图作为空间记忆；这是前代工作，不能直接作为 Atlas 具体实现。
- [How to use Atlas](https://atlas-beta.worldlabs.ai/docs/how-to-use-atlas) 与 [atlasGenerate API](https://atlas-beta.worldlabs.ai/docs/api-reference/atlas-generate)：明确由调用方提供目标相机。当前接口要求 RGBD 上下文，但文档区分消费深度与不消费深度的模型路由；接口字段不能用于推断预训练监督。
- [Images to posed RGBD](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images)：普通图像先估计相机和深度的产品前端。当前项目没有实现该前端。

需要区分名称相近的项目：

- Magic Leap 的 [Atlas: End-to-End 3D Scene Reconstruction from Posed Images](https://arxiv.org/abs/2003.10432) 是 2020 年的三维重建工作，不是 World Labs Atlas。
- Meta 的 [Atlas 检索增强语言模型](https://arxiv.org/abs/2208.03299)研究少样本语言任务，与本项目目标不同。
- 李飞飞参与的 [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](https://arxiv.org/abs/2503.15877)，ICCV 2025，研究用二维网格组织三维高斯并复用二维扩散模型。可作为三维生成补充阅读，不能当作 World Labs Atlas 的模型论文。

## 核心相关论文

年份采用首次预印本发表年份；会议版本另行注明。每篇均提供论文页面及 PDF 入口。

已下载的原文见 [本地论文目录](papers/README.md)，随 Git 提交；当前包含 CAT3D 和 BlenderProc2，其余条目仅提供在线链接。

### 相机控制与多视角生成

**1. CAT3D: Create Anything in 3D with Multi-View Diffusion Models**

Ruiqi Gao 等，2024。 [论文](https://arxiv.org/abs/2405.10314) · [本地 PDF](papers/cat3d-2405.10314.pdf) · [在线 PDF](https://arxiv.org/pdf/2405.10314) · [作者项目页](https://cat3d.github.io/)

给定若干参考图像及目标相机视角，多视角扩散模型生成相互一致的新视角，再使用重建方法获得三维表示。它最直接对应本项目的“参考帧与相机轨迹到新视角，再到三维”的任务链。重点阅读相机条件、多视角联合生成和生成视图的重建方法；两阶段流程可作基线，不等同于 Atlas 的统一模型。

**2. CameraCtrl: Enabling Camera Control for Text-to-Video Generation**

Hao He 等，2024；ICLR 2025 会议题名为 *CameraCtrl: Enabling Camera Control for Video Diffusion Models*。 [论文](https://arxiv.org/abs/2404.02101) · [PDF](https://arxiv.org/pdf/2404.02101) · [官方代码](https://github.com/hehao13/CameraCtrl)

通过相机轨迹参数化及相机控制模块，为视频扩散模型提供明确的视点控制。重点研究相机条件的编码与注入方式，作为比较“有相机条件”和“仅文本描述运动”的基线。它采用对现有视频模型增加控制模块的路线，不能替代对 Atlas 原生空间上下文的实现研究。

**3. CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models**

Rundi Wu 等，2024。 [论文](https://arxiv.org/abs/2411.18613) · [PDF](https://arxiv.org/pdf/2411.18613) · [作者项目页](https://cat-4d.github.io/)

以相机位姿与时间共同控制多视角视频生成，再优化可形变三维高斯表示。适合后续研究相机运动与物体运动的分离。静态资产轨迹视频只能支持空间生成实验；动态世界实验还需要物体动画或物理模拟数据。

### 自回归扩散与训练目标

**4. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion**

Boyuan Chen 等，2024。 [论文](https://arxiv.org/abs/2407.01392) · [PDF](https://arxiv.org/pdf/2407.01392) · [官方代码](https://github.com/buoyancy99/diffusion-forcing)

使用逐 token 独立噪声等级，将因果序列预测与扩散去噪结合。可用于理解连续帧的自回归生成、上下文噪声处理和长序列滚动误差。它提供训练与采样思路，并不是已确认的 Atlas 训练算法。

**5. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**

Xingchao Liu、Chengyue Gong、Qiang Liu，2022；ICLR 2023。 [论文](https://arxiv.org/abs/2209.03003) · [PDF](https://arxiv.org/pdf/2209.03003) · [官方代码](https://github.com/gnobitab/RectifiedFlow)

通过学习连接噪声分布与数据分布的流，实现连续数据生成。Atlas 官方明确提到 rectified flow，因此这是理解其生成目标的基础阅读。小模型实验可从噪声到图像或潜变量的速度场回归入手，具体调度和采样步数仍需独立确定。

**6. Scalable Diffusion Models with Transformers**

William Peebles、Saining Xie，2022；ICCV 2023。 [论文](https://arxiv.org/abs/2212.09748) · [PDF](https://arxiv.org/pdf/2212.09748) · [官方代码](https://github.com/facebookresearch/DiT)

以 Transformer 处理潜空间图像 patch，构成 Diffusion Transformer。重点阅读 patch token、条件调制和模型规模设计。它适合作为缩小版扩散骨干的参考，但原始 DiT 本身不提供多视角空间上下文与自回归视频机制。

### 三维几何与可渲染表示

**7. DUSt3R: Geometric 3D Vision Made Easy**

Shuzhe Wang 等，2023；CVPR 2024。 [论文](https://arxiv.org/abs/2312.14132) · [PDF](https://arxiv.org/pdf/2312.14132) · [官方代码](https://github.com/naver/dust3r)

从图像对回归 point maps，并将多对结果对齐到共同坐标系。适合研究逐像素三维输出、跨视图对齐和几何一致性，也可作为生成结果的几何检查工具。模型估计的几何应与渲染器产生的真值深度明确区分。

**8. VGGT: Visual Geometry Grounded Transformer**

Jianyuan Wang 等，2025；CVPR 2025。 [论文](https://arxiv.org/abs/2503.11651) · [PDF](https://arxiv.org/pdf/2503.11651) · [官方代码](https://github.com/facebookresearch/vggt)

从单张或多张图像直接预测相机、深度、点图和三维点轨迹等属性。适合作为开放几何基线，检查生成视频是否能恢复一致的场景结构。该论文对应原始 VGGT，不能与 Atlas 官方图表中的 VGGT-Ω 1B 版本混用。

**9. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**

Ben Mildenhall 等，2020；ECCV 2020。 [论文](https://arxiv.org/abs/2003.08934) · [PDF](https://arxiv.org/pdf/2003.08934) · [官方代码](https://github.com/bmild/nerf)

使用已知相机的图像优化连续辐射场，通过体渲染合成新视角。它提供相机射线、场景表示和新视角评测的基础，也可作为单场景拟合基线。Tiny NeRF 是已有视图的轻量示例，不能满足本项目自由设计三维相机轨迹的主数据需求。

**10. 3D Gaussian Splatting for Real-Time Radiance Field Rendering**

Bernhard Kerbl 等，2023；ACM TOG / SIGGRAPH 2023。 [论文](https://arxiv.org/abs/2308.04079) · [PDF](https://arxiv.org/pdf/2308.04079) · [官方代码](https://github.com/graphdeco-inria/gaussian-splatting)

通过优化三维高斯及可见性感知的 splatting 渲染器实现新视角渲染。适合研究三维结果的显式表示和交互展示。应区分“从生成视图优化出高斯场景”和“模型直接预测三维高斯”两种任务。

### 三维资产与合成数据生成

**11. Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items**

Laura Downs 等，2022；ICRA 2022。 [论文](https://arxiv.org/abs/2204.11918) · [PDF](https://arxiv.org/pdf/2204.11918) · [Google 官方介绍](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)

提供真实日常物体的带纹理三维扫描资产，适合选取少量物体在 Blender 中重新渲染。可用它补充人工建模资产的外观分布。

**12. Objaverse: A Universe of Annotated 3D Objects**

Matt Deitke 等，2022；CVPR 2023。 [论文](https://arxiv.org/abs/2212.08051) · [PDF](https://arxiv.org/pdf/2212.08051) · [官方数据与接口](https://github.com/allenai/objaverse) · [官方渲染脚本](https://github.com/allenai/objaverse-rendering)

提供类别丰富的三维资产及元数据，适合按物体 ID 下载小子集。重点借鉴资产过滤、归一化和 Blender 渲染流程；大规模原库并不是 mini 数据集，必须保存实际选取的 ID 与逐资产许可。

**13. Kubric: A Scalable Dataset Generator**

Klaus Greff 等，2022；CVPR 2022。 [论文](https://arxiv.org/abs/2203.03570) · [PDF](https://arxiv.org/pdf/2203.03570) · [官方代码](https://github.com/google-research/kubric)

结合 Blender 与物理模拟生成带丰富标注的合成数据。适合后续加入运动物体、遮挡、深度和实例标注。静态轨迹实验可先使用 Blender，动态物理数据再参考 Kubric。

补充工具资料：[BlenderProc2: A Procedural Pipeline for Photorealistic Rendering](https://joss.theoj.org/papers/10.21105/joss.04901)（[本地 PDF](papers/blenderproc2-joss.04901.pdf)）及[官方实现](https://github.com/DLR-RM/BlenderProc)，用于参考 RGB、几何标注与相机采样的工程流程。

### 新增补充：多模态与像素空间参数化

**14. Modality Forcing for Scalable Spatial Generation**

Bardienus Pieter Duisterhof 等，2026。[论文全文](https://arxiv.org/html/2606.13676v1) · [论文页面](https://arxiv.org/abs/2606.13676)。通过分模态噪声与解码器，将 T2I 模型后训练为联合 RGB／深度生成器，使用深度数据。它有相关作者，仍是独立工作，不能当作 Atlas 的完整配方或只用 RGB 训练的证明。本次在线阅读，未下载新 PDF。

**15. Back to Basics: Let Denoising Generative Models Denoise（JiT）**

Tianhong Li、Kaiming He，2025。[论文全文](https://arxiv.org/html/2511.13720v1) · [作者代码](https://github.com/LTH14/JiT)。论文区分网络直接预测的量与损失／采样所在空间：干净图像预测也可以转换成速度后做流匹配。它与当前去掉 VAE 的像素版有关，提示应单独验证预测参数化，而非仅凭 RF 公式成立就认定速度输出适合当前小模型。它不是 Atlas 的结构说明，也不能用其大 patch 实验直接认定本地代码根因。本次在线阅读，未下载新 PDF。

## 小型三维数据源选型

“mini”指本项目选出的固定小子集，而非 Atlas 官方发布的数据集。初期同时需要孤立物体与多物体场景：前者便于验证几何，后者才能覆盖遮挡、视差和空间移动。

| 数据源 | 建议首轮范围 | 主要用途 | 获取入口与注意事项 |
| --- | --- | --- | --- |
| Replica | 已选用；先取 apartment_0 验证，再增加独立场景 | 真实室内场景的相机轨迹与空间一致性 | [官方仓库](https://github.com/facebookresearch/Replica-Dataset)，非商业研究或教育用途；当前轻量渲染采用顶点色近似 |
| Khronos glTF Sample Assets | 先用已下载的 3 个资产，再扩至约 10 个 | 验证导入、材质、尺度与相机轨迹 | [官方仓库](https://github.com/KhronosGroup/glTF-Sample-Assets)，优先 GLB；逐模型保留许可与署名 |
| Kenney Furniture Kit | 选约 20 件家具，组合成约 10 个房间布局 | 推拉、横移、升降、绕障碍轨迹；构造真实视差 | [官方资产包](https://kenney.nl/assets/furniture-kit)，官方标注 CC0，下载时保留包内许可；外观偏低多边形 |
| Google Scanned Objects | 约 20 个不同类别物体 | 带真实扫描纹理的物体与桌面场景 | [官方介绍与下载入口](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)，全库 1030 个、约 13 GB、CC BY 4.0；仅取小子集 |
| Objaverse | 后续按许可和质量筛选约 30 个物体 | 扩充类别与材质分布 | [官方接口](https://github.com/allenai/objaverse)，按 ID 下载；检查纹理、缺失文件、重复资产与逐物体许可 |

以上数量均为实验建议。Khronos 是样例资产集合，Kenney 是游戏资产包，GSO 和 Objaverse 是研究数据集；它们的子集可统一转换为本项目的渲染输入，不能把三个物体称作三个数据集。

### 已下载的首批样本

| 资产 | 原始 GLB 大小 | 来源与许可摘要 | 本地文件 |
| --- | --- | --- | --- |
| Avocado | 8,110,040 字节 | Microsoft，CC0 | `data/raw/khronos/Avocado.glb` |
| ToyCar | 5,422,412 字节 | Guido Odendahl、Eric Chadwick，CC0 | `data/raw/khronos/ToyCar.glb` |
| Fox | 162,852 字节 | 模型 CC0；绑定、动画和转换 CC BY 4.0 | `data/raw/khronos/Fox.glb` |

合计 13,695,304 字节，约 13.06 MiB。固定来源提交、下载地址与 SHA256 见 [资产清单](asset-manifest.json)，原始署名见 [Avocado](asset-licenses/Avocado.md)、[ToyCar](asset-licenses/ToyCar.md)、[Fox](asset-licenses/Fox.md)。这三份文件属于同一个 Khronos 数据源，保留作早期导入调试；当前训练数据采用 Replica。Kenney、GSO、Objaverse 的子集尚未下载。

三个文件均已核对大小和 SHA256，并在 Blender 4.3.2 中成功导入，分别读入 1、3、2 个网格对象；材质数据随模型读入，最终外观仍需渲染抽查。

## 相机轨迹训练集建议

### 首轮规模与轨迹

先用 3 个样本验证导入和相机投影，再扩展为约 30 个资产或场景。首轮建议每个场景生成 6 类轨迹、每类 2 个种子；每段 48 帧、12 fps、256 × 256。30 × 6 × 2 = 360 段视频，共 17,280 帧、24 分钟。该规模用于小模型实验和流程验证，不能支撑 Atlas 级别的通用世界建模能力。

| 轨迹 | 相机运动定义 | 训练重点 |
| --- | --- | --- |
| 环绕 orbit | 围绕目标作圆弧平移，并持续朝向目标 | 视角变化、背面与遮挡关系 |
| 推拉 dolly | 沿观察方向移动，保持焦距不变 | 透视与尺度变化 |
| 横移 truck | 沿相机横轴移动，保持朝向不变 | 横向视差与新区域显露 |
| 升降 crane | 改变相机高度，轨迹内固定或平滑调整朝向 | 垂直视差与俯视结构 |
| 摇摄 pan 和 tilt | 相机中心固定，改变水平或俯仰角 | 单独学习旋转；须配合平移轨迹获得几何信息 |
| 复合轨迹 | 平滑组合平移、升降与转向，可沿样条移动 | 多动作组合与连续空间一致性 |

物体置于中心时，环绕可使用 look-at；房间中的横移与推拉不应全部强制看向同一物体。轨迹采样须检查相机是否穿模、目标是否持续出画，以及近远裁剪面是否合理。先固定场景内容和光照，只改变相机，避免把光照变化误当作视点变化。

### 每段数据的交付格式

训练以无损帧和逐帧参数为准；MP4 用于观看与抽查，避免把视频编码误差引入几何监督。

- `rgb/000000.png` 等连续图像，以及同帧率的 `preview.mp4`。
- `cameras.json`：每帧的 `frame_id`、`timestamp`、相机内参 `K`、`c2w`、`w2c`、图像尺寸、坐标系定义、裁剪面、轨迹类型及种子。
- `depth/`：浮点深度和有效掩码；明确记录相机坐标的 Z 深度或沿射线的距离，二者需要按射线方向转换，不能混用。
- `scene.json`：资产 ID、来源与许可、物体世界变换、归一化矩阵、光照参数、场景种子、渲染器版本和配置。
- 动态数据另存时间、动画动作与物体变换；有需求时再增加法线、实例分割和光流。

相机外参同时保留 Blender 原始约定和训练约定。若采用 OpenCV 相机坐标（右、下、前），从 Blender 相机坐标（右、上、后）转换时可用 `c2w_cv = c2w_blender @ diag(1, -1, -1, 1)`；世界坐标不随该转换改变。内参需考虑渲染分辨率、sensor fit 和像素宽高比，并用已知三维点重投影检查。

### 数据划分与训练样本

以资产或场景组划分训练、验证、测试集，建议约 80% / 10% / 10%，固定种子并保存 ID 清单。同一物体的不同轨迹、种子及含该物体的组合场景不能跨集合；可按共享资产形成的连通组划分，防止外观泄漏。另设在已知场景上保留轨迹的实验，单独评估轨迹泛化。

每个训练样本取 1 至 3 个参考视图及其相机参数，再给出目标轨迹，预测未作为参考输入的 RGB 帧；几何分支可同时预测深度。目标帧的真值 RGB 和深度只能用于监督，不能作为条件输入。先学习静态空间，再加入物体动画；仅有移动相机的视频不能证明模型学会了物理动态。

### 验证方法

下载阶段记录来源版本、逐文件哈希及许可；导入后检查模型、材质和包围盒。渲染后核对帧数与相机参数一一对应，抽查黑帧、缺失纹理、穿模、轨迹连续性，并验证相机矩阵互逆及已知点重投影误差。

有真值目标视图时，可记录 PSNR、SSIM 和 LPIPS；有深度时增加深度误差及跨视图重投影一致性。以固定场景 ID、参考视图数和轨迹做可比实验，分别报告单场景拟合与未见场景结果。

## 阅读与实验顺序

1. 阅读 Atlas 官方文章与 CAT3D，确定相机条件的新视角生成任务及缩小版复现边界。
2. 结合 CameraCtrl、NeRF、Kubric 和 BlenderProc2，落实三维资产到带标注轨迹视频的数据流程。
3. 结合 DiT、Rectified Flow、Diffusion Forcing，比较单目标视图生成与自回归序列生成。
4. 使用 DUSt3R 或 VGGT 检查几何一致性，再参考 3DGS 验证显式三维重建；静态流程稳定后研究 CAT4D。
