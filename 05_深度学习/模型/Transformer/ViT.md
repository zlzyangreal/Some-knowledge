# ViT

> Vision Transformer（Dosovitskiy et al., 2020）：把图像切成 patch 当 token，直接用标准 Transformer 编码器做分类。
> 它是后面所有视觉 Transformer 的起点（[[PVT]]、[[Swin Transformer]]、[[CSWin]]、[[SG-Former]] 都在它的基础上改）。

## 1. 一句话流程

```
图 224×224
 → 切 16×16 patch，共 (224/16)² = 196 个
 → 每个 patch 拉平成 768 维向量（16×16×3），线性投影成 token
 → 加 [CLS] token + 位置编码
 → 12 层 Transformer Encoder（MHSA + MLP）
 → 取 [CLS] 输出接 MLP 头分类
```

## 2. 关键设计

| 组件 | 作用 | 备注 |
| --- | --- | --- |
| Patch Embedding | 用 `kernel=16, stride=16` 的卷积实现 | 等价于切块 + 线性投影 |
| `[CLS]` token | 聚合全局信息，作为分类向量 | 也可改用全局平均池化 |
| 位置编码 | 补回 patch 的空间顺序 | 可学习（ViT 原版）或正弦 |
| MHSA | 全局注意力，一层就能看到全图 | 复杂度 O((HW)²)，是大图慢的原因 |
| MLP | 两层全连接 + GELU | 通道维度放大 4 倍 |
| LayerNorm | 放在注意力/MLP **之前**（Pre-LN） | 训练更稳，ViT 用 Pre-LN |

## 3. 为什么 ViT 一开始“不好训”

- 缺少卷积的归纳偏置（局部性、平移等变），必须靠**大数据**（ImageNet-21k / JFT-300M）学出来。
- 小数据上直接训 ViT 会不如 ResNet；解决办法是**预训练 + 微调**，或用 DeiT 的强增强 + 蒸馏。
- 现在工程上更常用**分层 ViT**（Swin/PVT），因为它们能输出多尺度特征，接检测/分割头方便。

## 4. 最小代码（PyTorch 自带实现）

```python
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)              # [1, 1000]
print(logits.argmax(dim=1))
```

手写 core 部分（理解用）：

```python
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, patch=16, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_c, dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)                    # [B, dim, H/p, W/p]
        return x.flatten(2).transpose(1, 2)  # [B, N, dim]
```

## 5. 部署到端侧要注意

1. 注意力的 `MatMul` 在很多 NPU 上不是最优算子，可能回落到 CPU 执行。
2. 输入分辨率一变，token 数平方级增长，延迟会爆；固定 224/256 输入最稳。
3. 导出 ONNX 时检查 `Reshape`/`Transpose` 的数量，它们是端侧的主要瓶颈。
4. 想又快又准：考虑 [[RepViT]] 这类“卷积主干 + Transformer 块”的混合结构。

## 相关

- [[什么是transformer]] — 注意力原理
- [[PVT]]、[[Swin Transformer]]、[[CSWin]]、[[SG-Former]]、[[RepViT]] — 改进版本
- [[transformer]] — 本目录总览
- [[三种创建的方式，在实际部署时它封装好的库是直接创建模型吗？]] — 部署视角
