# 1. Code
```bash
ultralytics/ultralytics/nn/modules/attention_module/
│── __init__.py
│── simam.py
│── ema.py   ← add EMA Code
```
ema.py（From the official source）
```python
"""
EMA (Efficient Multi-Scale Attention) Module for YOLO11
Stable version: supports H!=W, dynamic channels, safe for YOLO11 backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention Module

    Args:
        channels (int): number of input channels
        factor (int): group factor for channel grouping (default 32)
    """

    def __init__(self, channels, factor=32):
        super().__init__()
        self.channels = channels
        self.factor = factor

        # 延迟构建卷积层
        self.conv1x1_h = None
        self.conv1x1_w = None
        self.conv3x3 = None

    def _build_layers(self, c_per, device):
        """动态构建卷积层"""
        self.conv1x1_h = nn.Conv2d(c_per, c_per, kernel_size=1, stride=1, padding=0).to(device)
        self.conv1x1_w = nn.Conv2d(c_per, c_per, kernel_size=1, stride=1, padding=0).to(device)
        self.conv3x3 = nn.Conv2d(c_per, c_per, kernel_size=3, stride=1, padding=1).to(device)
        self.gn = nn.GroupNorm(num_groups=max(1, c_per // 8), num_channels=c_per)

    def forward(self, x):
        b, c, h, w = x.shape
        c_per = max(1, c // self.factor)

        # 首次 forward 动态构建卷积
        if self.conv1x1_h is None:
            self._build_layers(c_per, x.device)

        # 分组
        group_x = x.reshape(b * self.factor, -1, h, w)  # (B*g, c//g, H, W)

        # 水平与垂直 pooling
        x_h = torch.mean(group_x, dim=3, keepdim=True)  # (B*g, c_per, H, 1)
        x_w = torch.mean(group_x, dim=2, keepdim=True)  # (B*g, c_per, 1, W)

        # 1x1 卷积 + sigmoid
        attn_h = torch.sigmoid(self.conv1x1_h(x_h))
        attn_w = torch.sigmoid(self.conv1x1_w(x_w))

        # 上采样到原始 H,W
        attn_h = F.interpolate(attn_h, size=(h, w), mode='bilinear', align_corners=False)
        attn_w = F.interpolate(attn_w, size=(h, w), mode='bilinear', align_corners=False)

        # 应用注意力
        out = group_x * attn_h * attn_w
        out = self.gn(out)
        out = self.conv3x3(out)

        # 恢复原始 batch & channel
        return out.reshape(b, c, h, w)


```
`/home/alphafox/Desktop/Code/ultralytics/ultralytics/nn/modules/attention_module/__init__.py`
```python
from .ema import EMA
```
`/home/alphafox/Desktop/Code/ultralytics/ultralytics/nn/modules/__init__.py`
add
```python
from .attention_module.ema import EMA

__all__ =(
 ...
 EMA
 ...
)
```
/home/alphafox/Desktop/Code/ultralytics/ultralytics/nn/tasks.py
add code
```python
from ultralytics.nn.modules.attention_module.ema import EMA
```
# 2. Yaml
## 2.1 Backbone
```yaml
# Ultralytics YOLO11 object detection model with 3x EMA modules
# EMA is inserted after the last 3 C3k2 (C2f) blocks in the backbone

# number of classes
nc: 80

# model compound scaling constants
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

scale: s

# YOLO11 backbone with 3x EMA
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]          # 0 - P1/2 -> 64
  - [-1, 1, Conv, [128, 3, 2]]         # 1 - P2/4 -> 128
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2 -> 256
  - [-1, 1, Conv, [256, 3, 2]]         # 3 - P3/8 -> 256
  
  # --- P3 Stage ---
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4 -> 512 (Note: Scaled to 256 in 's' model)
  - [-1, 1, EMA, [256, 32]]            # 5 - EMA (P3 Attention) [Args: Channels, Factor]

  # --- Downsample to P4 ---
  - [-1, 1, Conv, [512, 3, 2]]         # 6 - P4/16 -> 512
  
  # --- P4 Stage ---
  - [-1, 2, C3k2, [512, True]]         # 7 -> 512
  - [-1, 1, EMA, [512, 32]]            # 8 - EMA (P4 Attention)

  # --- Downsample to P5 ---
  - [-1, 1, Conv, [1024, 3, 2]]        # 9 - P5/32 -> 1024
  
  # --- P5 Stage ---
  - [-1, 2, C3k2, [1024, True]]        # 10 -> 1024
  - [-1, 1, EMA, [1024, 32]]           # 11 - EMA (P5 Attention)

  - [-1, 1, SPPF, [1024, 5]]           # 12 - SPPF
  - [-1, 2, C2PSA, [1024]]             # 13 - C2PSA

# YOLO11 head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 14
  - [[-1, 8], 1, Concat, [1]]                    # 15 - concat with P4 EMA (idx 8)
  - [-1, 2, C3k2, [512, False]]                  # 16

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 17
  - [[-1, 5], 1, Concat, [1]]                    # 18 - concat with P3 EMA (idx 5)
  - [-1, 2, C3k2, [256, False]]                  # 19 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]                   # 20
  - [[-1, 16], 1, Concat, [1]]                   # 21 - concat with head P4
  - [-1, 2, C3k2, [512, False]]                  # 22 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]                   # 23
  - [[-1, 13], 1, Concat, [1]]                   # 24 - concat with backbone P5 (idx 13)
  - [-1, 2, C3k2, [1024, True]]                  # 25 (P5/32-large)

  - [[19, 22, 25], 1, Detect, [nc]]              # Detect(P3, P4, P5)

```
## 2.2 Neck
```python
# Ultralytics YOLO11 object detection model with EMA in Neck (PANet)
# EMA is inserted after every C3k2 block in the Head

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# YOLO11n backbone (Unchanged)
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]] # 4
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]] # 6
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]] # 8
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head with EMA
head:
  # --- Top-Down Path ---
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 11
  - [[-1, 6], 1, Concat, [1]]                    # 12 - cat backbone P4
  - [-1, 2, C3k2, [512, False]]                  # 13
  - [-1, 1, EMA, [512, 32]]                      # 14 - EMA added (P4 feature refinement)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]   # 15
  - [[-1, 4], 1, Concat, [1]]                    # 16 - cat backbone P3
  - [-1, 2, C3k2, [256, False]]                  # 17
  - [-1, 1, EMA, [256, 32]]                      # 18 - EMA added (P3 feature refinement)

  # --- Bottom-Up Path ---
  - [-1, 1, Conv, [256, 3, 2]]                   # 19
  - [[-1, 14], 1, Concat, [1]]                   # 20 - cat head P4 (Points to EMA layer 14)
  - [-1, 2, C3k2, [512, False]]                  # 21
  - [-1, 1, EMA, [512, 32]]                      # 22 - EMA added (P4 output)

  - [-1, 1, Conv, [512, 3, 2]]                   # 23
  - [[-1, 10], 1, Concat, [1]]                   # 24 - cat backbone P5 (Index 10 unchanged)
  - [-1, 2, C3k2, [1024, True]]                  # 25
  - [-1, 1, EMA, [1024, 32]]                     # 26 - EMA added (P5 output)

  # --- Detect ---
  - [[18, 22, 26], 1, Detect, [nc]]              # 27 - Detect(EMA_P3, EMA_P4, EMA_P5)
```