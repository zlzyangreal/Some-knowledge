# py
```python
"""
TransNeXt Aggregated Attention Module for YOLO11 (STABLE ENGINEERING VERSION)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Convolutional GLU
# -------------------------------------------------
class ConvolutionalGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, kernel_size=3):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2

        self.norm = nn.BatchNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim * 2, 1)
        self.dwconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size,
            padding=kernel_size // 2, groups=hidden_dim
        )
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x, gate = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x)
        x = x * self.act(gate)
        x = self.fc2(x)
        return x


# -------------------------------------------------
# Aggregated Attention (YOLO-safe)
# -------------------------------------------------
class AggregatedAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        window_size=7,
        pool_size=7,
        qkv_bias=True,
        proj_drop=0.,
        attn_drop=0.
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.pool_size = pool_size
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_local = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.pool = nn.AdaptiveAvgPool2d(pool_size)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        q = self.q(x).reshape(
            B, H * W, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3) * self.scale

        x_2d = x.reshape(B, H, W, C)

        k_local, v_local = self._sliding_window_kv(x_2d)
        k_global, v_global = self._pooled_kv(x_2d)

        k = torch.cat([k_local, k_global], dim=3)
        v = torch.cat([v_local, v_global], dim=3)

        k = k.reshape(B, self.num_heads, -1, self.head_dim)
        v = v.reshape(B, self.num_heads, -1, self.head_dim)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x.transpose(1, 2).reshape(B, C, H, W)

    def _sliding_window_kv(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        pad = ws // 2

        x_pad = F.pad(
            x.permute(0, 3, 1, 2),
            (pad, pad, pad, pad),
            mode='reflect'
        ).permute(0, 2, 3, 1)

        x_unfold = x_pad.unfold(1, ws, 1).unfold(2, ws, 1)
        x_unfold = x_unfold.reshape(B, H * W, ws * ws, C)

        kv = self.kv_local(x_unfold).reshape(
            B, H * W, ws * ws, 2, self.num_heads, self.head_dim
        ).permute(3, 0, 4, 1, 2, 5)

        return kv[0], kv[1]

    def _pooled_kv(self, x):
        B, H, W, C = x.shape

        x_pooled = self.pool(
            x.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)

        x_pooled = x_pooled.reshape(B, 1, self.pool_size * self.pool_size, C)
        x_pooled = x_pooled.expand(B, H * W, self.pool_size * self.pool_size, C)

        kv = self.kv_global(x_pooled).reshape(
            B, H * W, self.pool_size * self.pool_size, 2, self.num_heads, self.head_dim
        ).permute(3, 0, 4, 1, 2, 5)

        return kv[0], kv[1]


# -------------------------------------------------
# AA-SPPF (YOLO Neck replacement)
# -------------------------------------------------
class AA_SPPF(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        k=5,
        num_heads=8,
        window_size=7,
        pool_size=None
    ):
        super().__init__()

        pool_size = pool_size or k
        c_ = c1 // 2
        assert c_ % num_heads == 0

        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.aa = AggregatedAttention(
            dim=c_,
            num_heads=num_heads,
            window_size=window_size,
            pool_size=pool_size
        )
        self.cv2 = nn.Conv2d(c_, c2, 1, 1)

    def forward(self, x):
        return self.cv2(self.aa(self.cv1(x)))

```
# yaml(将 SPPF 模块替换为 **Aggregated Attention (AA)** 模块)
```yanl
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs

# Parameters
nc: 80
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# ---------------- Backbone ----------------
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]

  # ---------- SPPF → Aggregated Attention ----------
  - [-1, 1, AA_SPPF, [1024, 5]]
  #            ↑    ↑  ↑  ↑
  #           c2  heads win pool


  - [-1, 2, C2PSA, [1024]]             # 10

# ---------------- Head ----------------
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]         # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]         # 16 (P3/8)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]         # 19 (P4/16)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]         # 22 (P5/32)

  - [[16, 19, 22], 1, Detect, [nc]]

```
# task add
```python
	base_modules = frozenset(
        {
            # add
            AA_SPPF,
            AggregatedAttention,
            ConvolutionalGLU,
```