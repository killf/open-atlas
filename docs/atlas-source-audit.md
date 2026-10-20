# Atlas 一手资料核对与当前实现边界

核对日期：2026-10-20（北京时间）。对应代码：`1837b8a`。本次在线阅读官方技术文章、当前 Beta API、相关原论文和作者实现；产品接口说明与研究模型训练机制分别核对。以下“未公开”表示本次在所列来源中未找到可确认的细节，不是证明其他地方不存在。

## 核对结论

**相机条件有官方依据，应保留。** 当前任务可写为：

`历史 RGB + 对应相机 + 目标相机 + 视频时间差 → 下一帧 RGB`

这属于相机控制的新视角生成。用户只上传普通图像的产品流程，可以先估计输入相机，再指定目标相机生成；不能据此推断生成网络不使用相机。当前项目没有图像位姿估计前端，因此普通无位姿 MP4 不能直接接入。

**当前版本是 RGB 像素空间的小规模实验。** Atlas 官方明确将研究模型描述为潜空间、多模态、自回归 rectified-flow Transformer；我们去掉编解码器，只保留 RGB 任务，是已明确选择的简化。具体相机编码、块结构、上下文选择和训练配方不能用相关论文补全后宣称为 Atlas 原始设计。[S1]

## 官方来源：产品流程与研究模型

| 编号 | 一手来源及定位 | 能确认的事实 | 不能据此推出的结论 |
| --- | --- | --- | --- |
| S1 | [Atlas 官方文章](https://www.worldlabs.ai/blog/atlas)，Model Architecture、Benchmarks，2026-09-01 | 研究模型具有潜空间、多模态、自回归及 rectified flow 机制 | 不知道其层数、通道数、相机编码器、噪声调度或完整训练模态配比 |
| S2 | [RTFM 官方文章](https://www.worldlabs.ai/blog/rtfm)，Scalability、Persistence，2025-10-16 | 前代模型按目标位姿查询下一帧；带位姿历史帧形成空间记忆，并检索附近视图 | 不能断言 Atlas 原样沿用 RTFM 的具体缓存和检索策略 |
| S3 | [How to use Atlas](https://atlas-beta.worldlabs.ai/docs/how-to-use-atlas)，Pose your images、Generate new views | 产品流程先构建带相机的视图；应用指定 `targetCameras`，也可把生成视图继续作为上下文 | 不能把“只上传 RGB”理解为生成阶段没有相机条件 |
| S4 | [Images to posed RGBD](https://atlas-beta.worldlabs.ai/docs/images-to-posed-images)，Choose your inputs、Read the result | `images2PosedRGBD` 接收普通图像，估计相机与深度；文档描述单视图／多视图后端路径 | 不能证明该服务的每个步骤均由同一个 Atlas 网络端到端完成 |
| S5 | [atlasGenerate API](https://atlas-beta.worldlabs.ai/docs/api-reference/atlas-generate)，Request body；[操作指南](https://atlas-beta.worldlabs.ai/docs/atlas-walkthrough) | 当前接口要求 `contextFrames` 与 `targetCameras`；每个上下文含图像、相机和深度。文档又明确区分使用深度的 warp 路由与不消费深度的 hero base 路由 | API 字段必填不等于所有模型路由都使用该模态，也不能反推预训练只用 RGB 或必须联合监督 RGBD |
| S6 | [Posed RGBD generation](https://atlas-beta.worldlabs.ai/docs/atlas-generate)，Request output depth | `returnDepth` 描述对生成视图重建深度的产品流程 | 不能据响应里有深度，就断言该路由的生成网络直接输出四通道 RGBD |
| S7 | [Cameras and posed images](https://atlas-beta.worldlabs.ai/docs/cameras-and-posed-images)，Camera model、Resizing | 位姿是 camera-to-world；RDF 为右／下／前；缩放图像须同步缩放内参 | 接口格式并未公开 Atlas 内部相机 token 或射线编码方式 |

Beta 文档为本次查询时的服务状态，并非 2026-09-01 研究模型架构的逐项披露。API 中的 depth 字段、某个推理路由是否读取深度、训练是否使用深度监督，是三个不同问题。当前用户约束仍是深度不参与本项目训练。

## 相关原论文和公开代码

这些来源用于核对算法与实现选择，不代表它们就是 Atlas 的训练实现。

| 编号 | 原文与定位 | 与本项目的关系 |
| --- | --- | --- |
| P1 | [Rectified Flow](https://arxiv.org/html/2209.03003v1)，§2.1、式 (1)；[作者损失代码](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/losses.py)、[采样代码](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/sampling.py) | 当前线性插值、速度 MSE 与 Euler 积分的依据 |
| P2 | [DiT](https://arxiv.org/html/2212.09748v2)，§3、附录 A；[作者 models.py](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py) | adaLN-Zero、patch 投影与输出还原的参考；当前不等于原版 DiT，更不能据此确认 Atlas 的块结构 |
| P3 | [CameraCtrl](https://arxiv.org/html/2404.02101v1)，§3.2–3.3；[作者 ray_condition](https://github.com/hehao13/CameraCtrl/blob/af45a7a0dbeedb7f6ed66aac4a9e5411a279f0d2/cameractrl/data/dataset.py) | 六维 Plücker 射线的参考；当前没有实现其独立相机编码器与多尺度融合 |
| P4 | [CAT3D](https://arxiv.org/abs/2405.10314)，§3.1；[已提交 PDF](papers/cat3d-2405.10314.pdf) | 第一参考帧相对坐标、图像通道拼接射线的参考；CAT3D raymap 为原点与方向，不能与 Plücker moment／方向等同 |
| P5 | [Diffusion Forcing](https://arxiv.org/html/2407.01392v3)，§3、附录 B.6 | 各时刻独立噪声水平区别于干净历史的 teacher forcing；当前没有实现 DF，亦未确认 Atlas 采用 DF |
| P6 | [Modality Forcing for Scalable Spatial Generation](https://arxiv.org/html/2606.13676v1)，§3 | 相关作者的独立 RGB／深度联合生成工作，具有分模态噪声和解码器；使用深度数据，不能当作“Atlas 只用 RGB 训练”的证据，也不是 Atlas 完整训练配方 |
| P7 | [JiT: Back to Basics](https://arxiv.org/html/2511.13720v1)，§3、§4.2–4.3；[作者 denoiser.py](https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/denoiser.py) | 像素 Transformer 可直接预测干净图像，再转为速度进行流匹配和采样；说明预测参数化需要单独验证。它不是 Atlas 论文，论文中的大 patch 失效结论不能直接套用本地小 patch |

本次阅读 RF、DiT、CameraCtrl 与 JiT 作者仓库时固定上述提交。临时下载代码仅用于核验，没有拷贝进项目，也没有下载新的论文 PDF；原有 PDF 与清单保留。

## 对照本地代码

| 项目 | 本地实现 | 核对结果 |
| --- | --- | --- |
| 对外条件 | `data.py`、`sample.py` 读取历史 RGB、相机及目标相机 | 与 S3/S5 的相机控制任务方向一致；本地缺少 S4 的无位姿输入前端 |
| 网络输出 | `model.py::forward` 输出三通道速度，`sample_flow` 积分为 RGB | 与 P1 的速度参数化一致；不能把 `forward` 输出直接当图像 |
| 流方向 | `x_t=(1-t)*ε+t*I`，监督 `I-ε`，从 t=0 向 1 采样 | 与 P1 作者实现同向。有的论文采用相反 t 约定，移植时必须同时调整速度符号及积分方向 |
| 图像表示 | 直接在 RGB 像素上加噪和积分，无独立编解码器 | 与 S1 的潜空间表示不同，是本项目简化 |
| 相机射线 | `[d, o×d]`，以第一参考帧为坐标原点 | Plücker 定义正确；P3 作者代码顺序是 `[o×d, d]`，只差固定通道置换。相对坐标借鉴 P4；不能称为 Atlas 已公开的原生相机格式 |
| Transformer | 4 层、128 维、adaLN-Zero 风格、学习位置向量 | P2 是参考。当前位置编码、时间嵌入、初始化均有自身选择；例如最终调制层未照搬原版 DiT 的零初始化。没有证据表明这些选择就是 Atlas 配置 |
| 自回归上下文 | 固定选首帧和最近帧，干净历史仅条件化下一目标 | 是条件下一帧训练；没有实现统一的多模态序列或 P5 的独立历史噪声 |
| 长时记忆和缓存 | 未实现空间检索或 KV-cache；历史 token 相互可见，时间差还随目标改变 | 不具备 S2 的长期空间记忆能力；当前不能直接把历史状态当作可追加的因果 KV-cache 复用。Atlas 具体机制仍未知 |
| 几何输出 | 无深度头、点图头或三维表示输出 | 与当前 RGB-only 任务一致，不能宣称复现完整 Atlas 的几何能力 |

本次将公开 CameraCtrl 的 `ray_condition` 与本地 `geometry.rays` 在一个含旋转和平移的 16×16 相机实例上直接比较，置换通道后最大差异为 `2.96e-8`。这只验证该射线实例，不是 Atlas 架构相同或相机控制已经学好的证明。

P1 的作者实现与当前代码均使用目标减噪声的速度。网络额外接收加噪目标、输出速度是流匹配的内部接口；生成任务的最终输出仍是下一帧 RGB。网络也可使用干净图像预测参数化：令 `I_hat=net(x_t,t,context)`，再以 `v_hat=(I_hat-x_t)/(1-t)` 计算速度；接近 t=1 时需明确分母下限与相应训练定义。P7 作者代码就是先预测图像，再在速度空间计算损失与积分。这仍是多步生成，不等于无噪声输入的一步 RGB 回归。

**必须收紧此前结论：公式成立不代表当前输出参数化已经适合像素小模型。** P7 对像素 Transformer 的研究支持优先对照验证图像预测与速度预测。当前 4×4 RGB patch 为 48 维、隐藏层 128 维，并不存在该论文大 patch 超过隐藏维度的同类直接维度限制；尚不能以 P7 判定本模型低质量的根因。Atlas 原文未披露其网络究竟直接预测图像、速度还是其他等价参数化。

## 应修正的说法与后续边界

- “相机条件没有 Atlas 依据”不成立：S2/S3/S5 明确支持该任务。
- “Rectified Flow 必须由网络直接输出速度”不成立：P7 展示了图像预测转速度的实现。当前速度输出的公式可以成立，其优化效果仍需验证。
- “普通 RGB 可作为用户输入，所以模型训练必定只用 RGB”不成立：用户输入、服务预处理与训练监督须分别核实。
- 不再把 API 深度字段或联合几何能力等同于本项目必须拼接 RGBD 训练；本地深度训练仍保持关闭。
- 当前只能称为参考 Atlas 公开机制的 RGB 像素空间实验，不能声称完整结构复现。完整潜空间方案与当前无编解码器简化须明确区分。
- 本次未找到可用来完整复现 World Labs Atlas 的公开训练代码、权重或独立完整配方。相关论文与同名仓库均不能替代这一缺失。
- S2 展示了无需显式三维表示的学习式渲染；缺少手工重投影或显式几何模块，本身不能认定为架构错误。
- 本次仅修正文档，不以资料检索代替质量诊断。此前真实样本单步效果仍差；继续训练前应先完成小样本过拟合验证。

## 同名项目辨别

[Meta Atlas](https://github.com/facebookresearch/atlas) 是检索增强语言模型；[Magic Leap Atlas](https://arxiv.org/abs/2003.10432) 是 posed-image 三维重建；[Gaussian Atlas](https://arxiv.org/abs/2503.15877) 是二维扩散与三维高斯研究。它们均不是本项目所指的 World Labs Atlas。P6 虽有相关作者，也必须作为独立论文引用。
