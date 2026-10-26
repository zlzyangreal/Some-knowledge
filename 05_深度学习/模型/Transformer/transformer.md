# 1.简介
[[什么是transformer]]
# 2.网络架构
## 1.[[PVT]]
# #.transformer改
## 1.[[Swin Transformer]]
改进版[[CSWin]]
**超越前两者[[SG-Former]]**
## 2.1[[RepViT]]

## 3. 视觉 Transformer 家族怎么记

| 变体 | 一句话 | 属于 |
| --- | --- | --- |
| ViT | 把图切成 16×16 patch 当 token，纯 Transformer，需要大数据/强增强 | 平铺型 |
| PVT | 金字塔结构，多尺度特征，可接检测头 | 分层型 |
| Swin | 窗口内自注意力 + 窗口移位，复杂度线性 | 分层型 |
| CSWin | 十字形窗口（横条 + 竖条），扩大感受野 | 分层型 |
| SG-Former | 基于显著性的粒度自适应（重要区域用细粒度 token） | 分层型 |
| RepViT | 把 MobileNet 的块改造成 Transformer 风格（结构重参数化） | 轻量型 |

## 4. 与 CNN 的关键差异

| 维度 | CNN | ViT |
| --- | --- | --- |
| 感受野 | 逐层扩大 | 一层内全局 |
| 归纳偏置 | 强（局部性、平移等变） | 弱，需要更多数据 |
| 训练成本 | 低 | 高（要长训 + 强增强 + 大 batch） |
| 小数据表现 | 好 | 差，通常要预训练微调 |
| 部署友好度 | 高（算子规整） | 中（`Reshape`/`Softmax`/`MatMul` 多，端侧推理慢） |

## 5. 端侧部署注意

1. 注意力里的 `MatMul` 在高通/瑞芯微 NPU 上不一定有最优实现，可能回落到 CPU。
2. `Reshape`/`Transpose` 过多会触发内存搬运，是端侧 Transformer 慢的主因。
3. 混合架构（CNN 提取 + 轻量注意力）通常比纯 Transformer 更适合嵌入式。

## 相关

- [[什么是transformer]] — 原理解释
- [[PVT]]、[[Swin Transformer]]、[[CSWin]]、[[SG-Former]]、[[RepViT]]
- [[注意力机制索引]] — 模块级注意力
- [[网络主干概括]] — CNN 侧对照
