![[TransNeXt0.png]]
# 1. 背景与动机 (Background & Motivation)
这一步探讨 TransNeXt 想要解决的核心矛盾：**如何在降低计算量的同时，获得像人眼一样自然且全局的视觉感知？**
## 1.1 现有 ViT 的两大隐疾
虽然 Vision Transformer (ViT) 和其变体（如 Swin Transformer）很强大，但作者指出它们存在两个关键缺陷：
- **深度退化效应 (Depth Degradation)** 高效的 ViT 模型通常通过堆叠很多层来间接实现信息的全局混合。但研究表明，由于残差连接（Residual Connection）的存在，深层网络往往表现得像是浅层网络的“集成” 。这意味着，单纯靠“堆层数”并不一定能实现预期的充分信息混合，导致深层特征提取效率下降 。
- **非自然的视觉感知 (Unnatural Visual Perception)** 为了降低计算复杂度，许多模型（如 Swin）采用了**基于窗口（Window-based）**的注意力机制。
    - **问题**：这种把图片切成一个个小方块的做法，导致窗口中心的像素和边缘的像素被区别对待 。
    - **后果**：即便堆叠多层，这种人为的“割裂感”依然会留下的痕迹，导致模型产生的感受野（Receptive Field）出现不自然的伪影 。
## 1.2 图解分析：机器视觉 vs. 生物视觉
![[TransNeXtfigure2.png]]
**Figure 2 左侧：视觉机制对比**
- **Local Attention (最左)**：视野被死死限制在一个方框内，黑框区域完全不可见，这是典型的局部注意力缺陷 。
- **Sliding Window (左二)**：虽然试图通过移动窗口来缓解，但本质上还是受限的 。
- **Biological Vision (最右)**：这是人眼的**中央凹（Foveal）**视觉系统。红星代表注视点，该点周围极其清晰，而远处虽然模糊但依然能被感知到 。最重要的是，这种感知是随着眼球转动平滑移动的，对每一个像素都是公平的（平移等变性） 。
- **Pixel-focused Attention (TransNeXt, 右二)**：这是作者提出的方案。你看它的视野形态（亮色区域）与最右侧的生物视觉惊人地相似——**中心聚焦，并拥有全局的“余光”** 。
**Figure 2 右侧：有效感受野 (ERF) 可视化**
- **Swin / ConvNeXt**：看右侧的绿色热力图，Swin 的感受野呈现出明显的“十字形”或方块状，这证明了窗口划分带来了不自然的伪影。
- **TransNeXt**：其感受野呈现出完美的圆形高斯分布，这意味着它像人眼一样自然地聚焦于物体，没有人工切割的痕迹 。
## 1.3 核心灵感：仿生学设计
基于上述观察，TransNeXt 的设计哲学是**仿生（Biomimetic）**： 作者希望设计一种 Token Mixer（Token 混合器），它不需要依赖深层堆叠就能实现充分的信息混合，并且能模拟人眼的**连续眼动（Continuous Eye Movement）** 。
这就引出了论文的核心算法——**聚合注意力（Aggregated Attention）**，它旨在赋予每个像素（Token）这种“既有精细局部，又有广阔全局”的能力。
# 2. 核心算法设计 - 聚合注意力 (Aggregated Attention)
聚合注意力并不是一蹴而就的，它是基于基础版 **Pixel-focused Attention (PFA)** 逐步增强而来的
## 2.1 基础形态：Pixel-focused Attention (PFA)
为了模仿人眼的“中央凹”（清晰）与“余光”（模糊但广阔），作者设计了一种**双路**（Dual-path）并行机制。对于特征图上的每一个像素（查询 Query），它同时通过两条路径去“看”世界：
- **路径一：滑动窗口注意力 (Sliding Window Attention)**
    - **作用**：模拟**中央凹 (Fovea)**。
    - **机制**：查询像素 $Q_{(i,j)}$ 只与它周围 $k \times k$ 窗口内的近邻像素 $\rho(i,j)$ 互动。
    - **特点**：捕捉高频、精细的局部特征。
- **路径二：池化注意力 (Pooling Attention)**
    - **作用**：模拟**周边视觉 (Peripheral Vision)**。
    - **机制**：对整个特征图进行下采样（Pooling）得到概括性特征 $\sigma(X)$。查询像素 $Q_{(i,j)}$ 与这些概括性特征互动。
    - **特点**：虽然模糊（分辨率低），但覆盖了全局信息，且计算量很小。

关键创新：联合 Softmax
不同于以往的模型（如 Twins）把这两步分开做，TransNeXt 将这两条路径的相似度分数（Attention Scores）拼在一起，放入同一个 Softmax 中竞争。这意味着“局部细节”和“全局背景”会根据重要性动态分配注意力权重。
数学公式表达：
假设当前像素位置为 $(i,j)$，局部邻域为 $\rho(i,j)$，全局池化特征为 $\sigma(X)$：
$$S_{local} = Q_{(i,j)}K_{\rho(i,j)}^{T}$$
$$S_{global} = Q_{(i,j)}K_{\sigma(X)}^{T}$$
通过拼接 (Concat) 后计算 Softmax：
$$A = \text{Softmax}(\text{Concat}(S_{local}, S_{global}))$$
最终输出是对两条路径 Value (V) 的加权求和 。
## 2.2 进阶形态：聚合多样性注意力 (Aggregating Diverse Attentions)
作者发现，标准传的 QKV (Query-Key-Value) 注意力还可以进一步进化，于是引入了两个增强模块，形成了最终的 **Aggregated Attention (AA)**。
- **Query Embedding (查询嵌入)**
    - **灵感**：来自多模态模型（如 BLIP），用可学习的 Query 来提取特定信息。
    - **做法**：在原有 Query $Q_{(i,j)}$ 的基础上，直接加上一个可学习的向量 $QE$。
    - 公式：
$$S = (Q_{(i,j)} + QE)K^{T}$$
    - **意义**：这让模型不仅根据输入图像内容（Input-dependent）去关注，还能根据任务目标（Task-specific）去主动寻找特征。
- **Positional Attention (位置注意力)**
    - **问题**：传统的相对位置编码（Relative Position Bias）是静态的，不管输入图片长什么样，位置偏置都一样。
    - **做法**：引入一组可学习的 Token $T$，让它与当前的 Query 互动，生成动态的位置偏置。
    - 公式：
$$\text{PFA}(X) = (A_{local} + Q_{(i,j)}T)V_{local} + \dots$$
    - **意义**：这是一种 **Query-Learnable-Value (QLV)** 模式，它让位置感知变得动态化、数据驱动化，比死板的加性偏置更灵活。
## 2.3 图解分析：PFA vs. AA
![[TransNeXtfigure3.png]]
**左图 (Pixel-focused Attention)**：
- 你可以清晰地看到两条路：上方蓝色框是 **Sliding Window**（看局部猫眼），下方黄色框是 **Spatial Reduction**（看全局）。
- 两者在 Softmax 处汇合。
**右图 (Aggregated Attention)**：
- 结构类似，但在红色 Query 处增加了粉色的 **Query Embedding**。
- 在右上角增加了绿色的 **Learnable Tokens** 用于计算 Positional Attention。
- 这就是 TransNeXt 的完全体 Token Mixer。
## 2.4 解决多尺度问题：Length-scaled Cosine Attention
传统注意力用的是点积（Dot Product），但在处理大分辨率图片（Token 数量剧增）时，注意力分布会变得极其尖锐（Entropy 下降），导致外推性能差。
- **解决方案**：改用**余弦相似度 (Cosine Similarity)**，并引入与序列长度 $N$ 相关的缩放因子。
- 公式：
$$\text{Attention}(Q,K,V) = \text{Softmax}(\tau \log N \cdot \hat{Q}\hat{K}^{T})V$$
    其中 $\hat{Q}, \hat{K}$ 是归一化后的向量。
- **效果 (Figure 6 右图)**：
    图表显示，当输入分辨率从 224 增加到 640 时，使用了 **Length-scaled Cosine (红线)** 的模型准确率保持稳定，而普通模型（蓝线/灰线）性能大幅下降。这证明了该设计让模型能轻松适应不同大小的图片输入。
![[TransNeXtfigure6.png]]
# 3.通道混合器 - 卷积 GLU (Convolutional GLU)
在 Vision Transformer (ViT) 中，通常由 Token Mixer（如刚才讲的 Attention）负责空间信息交流，而 Channel Mixer（通常是 MLP/FFN）负责通道特征的混合。TransNeXt 对此进行了重新思考，提出了 **Convolutional GLU (ConvGLU)**。
## 3.1 设计动机：为什么传统的 MLP 不够好？
- **缺乏通道注意力 (Lack of Channel Attention)**：标准的 ViT 结构中的 FFN (Feed-Forward Network) 只是简单的线性映射和激活，缺乏对通道重要性的动态筛选能力 。
- **SE 机制的局限 (Limitations of SE)**：传统的 SE (Squeeze-and-Excitation) 模块虽然引入了通道注意力，但它依赖 **全局平均池化 (Global AvgPooling)** 来获取控制信号 。
    - **问题**：这会导致全图所有的 Token 共享同一个门控信号，粒度太粗 (Coarse-grained) 。在 ViT 时代，我们已经有了强大的全局 Attention，不需要这种粗糙的全局信号。
- **位置信息的缺失**：普通的 FFN 无法提供位置信息。
## 3.2 解决方案：Convolutional GLU
作者结合了 **Gated Linear Unit (GLU)** 的门控特性和 **卷积 (Convolution)** 的局部特性，设计了 ConvGLU。
- **核心结构**：它包含两个分支。
    - **值分支 (Value Branch)**：普通的线性投影。
    - **门控分支 (Gating Branch)**：在激活函数之前，插入了一个 **$3 \times 3$ 的深度卷积 (Depthwise Convolution)** 。
- **工作原理**：
    1. 门控分支通过 $3 \times 3$ 卷积感知**局部邻域**信息。
    2. 生成每个 Token 独有的门控信号（不再是全图共享）。
    3. 用这个信号去控制（点乘）值分支的输出。
数学表达：
假设输入为 $X$，输出为 $Y$：
$$Y = \text{Linear}_2(\text{Act}(\text{DWConv}(\text{Linear}_{g}(X))) \odot \text{Linear}_{v}(X))$$
其中 $\odot$ 表示逐元素相乘（Element-wise Multiplication）。
## 3.3 图解分析：Channel Mixer 对比
![[TransNeXtfigure4.png]]
1. **Original Feed-Forward (最左)**：标准的 `Linear -> Activation -> Linear`。结构简单，无门控。
2. **Convolutional Feed-Forward (左二)**：引入了卷积，但没有门控机制。
3. **Gated Linear Unit (中间)**：引入了门控（两条路合并），但通常是点对点的，缺乏空间交互。
4. **FFN with SE module (右二)**：注意那个蓝色的框 `Global AvgPool`。它把空间信息压缩成一个点，导致丢失了局部细节。
5. **Convolutional GLU (最右，Ours)**：
	- 采用了 GLU 的双路结构。
	- 在左侧门控通路中加入了粉色的 **DW Conv3x3**。
	- **优势**：这赋予了每个 Token 基于其**最近邻居 (Nearest Neighbors)** 的通道注意力，既增强了鲁棒性，又补充了位置信息 (CPE) 。
## 3.4 核心优势
1. **细粒度注意力**：每个像素都有自己的通道门控信号，比 SE 机制更精细。
2. **隐式位置编码**：$3 \times 3$ 卷积作为一个条件位置编码 (CPE)，让模型在不需要显式位置嵌入的情况下也能感知空间结构 。
3. **鲁棒性增强**：这种基于局部特征的筛选机制，被证明能有效提升模型对干扰的鲁棒性（在 ImageNet-A 等数据集上表现优异。
# 4. 整体架构与全篇总结 (Architecture & Summary)
## 4.1 TransNeXt 的宏观架构
TransNeXt 采用了一种**分层**（Hierarchical）结构，类似于 ResNet 或 PVT，这使它非常适合作为下游任务（如检测、分割）的主干网络。
![[TransNeXtfigure5.png]]
**分阶段设计 (4 Stages)**：
- 图像通过 **Patch Embedding** 进入网络。
- **Stage 1 - Stage 3**：这是核心创新区。堆叠了多个 Block，每个 Block 由 **Aggregated Attention** (Token Mixer) 和 **Convolutional GLU** (Channel Mixer) 组成 。注意特征图分辨率在每个阶段之间降低（$H/4 \to H/8 \to H/16$）。
- **Stage 4**：由于此时特征图已经很小（$H/32$），继续做空间池化意义不大。因此，作者在这里切回了标准的 **Multi-Head Self-Attention (MHSA)**，但依然保留了 Length-scaled Cosine Attention 和 Query Embedding 的增强特性 。
## 4.2 总结
TransNeXt 是一篇典型的“反思型”论文。它没有盲目堆砌层数，而是反思了 Vision Transformer 在迁移到视觉任务时的**生物学合理性**。通过 **Aggregated Attention** 模拟人眼变焦与余光，以及 **Convolutional GLU** 增强局部鲁棒性，它在无需极深网络的情况下，实现了极其自然且鲁棒的视觉特征提取。
### 魔改建议
将 TransNeXt 的机制（特别是聚合注意力和 ConvGLU）引入 YOLO 系列模型，**有极大的概率能提高检测精度（mAP）和鲁棒性**，特别是对于**小目标、遮挡物体**以及**多尺度变化**的场景。
方案 A：替换 SPP/SPPF 模块 (最推荐)
YOLO 的 Neck 部分通常有一个 SPP (Spatial Pyramid Pooling) 模块用于扩大感受野。
- **操作**：将 SPPF 模块替换为 **Aggregated Attention (AA)** 模块。
- **理由**：AA 本身就包含“池化路径” ，天然具备 SPP 的功能，但多了动态的注意力加权。这能以极小的代价显著提升深层特征的全局感知。
方案 B：改造 C2f / C3 模块的 Bottleneck
YOLOv8 使用 C2f，YOLOv5 使用 C3。它们内部的核心是 `Conv -> Conv`。
- **操作**：将 C2f 中的 Bottleneck 结构替换为 TransNeXt Block。
    - **Token Mixer**: 使用 Aggregated Attention。
    - **Channel Mixer**: 使用 **Convolutional GLU** 替换原有的普通卷积层 。
- **位置建议**：仅在 Backbone 的最后两个 Stage (P4, P5) 进行替换。浅层保持纯卷积以维持高 FPS，深层用 Attention 提炼语义。