# 1. Code
```bash
ultralytics/ultralytics/nn/modules/attention_module/
│── __init__.py
│── simam.py
│── cbam.py   ← add CBAM Code
```
## 1.1 CBAM Code
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                              padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn: x = self.bn(x)
        if self.relu: x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None

        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.avg_pool2d(x, x.size()[2:])
            elif pool_type == 'max':
                pool = F.max_pool2d(x, x.size()[2:])
            elif pool_type == 'lp':
                pool = F.lp_pool2d(x, 2, x.size()[2:])
            elif pool_type == 'lse':
                pool = logsumexp_2d(x)

            channel_att_raw = self.mlp(pool)

            channel_att_sum = channel_att_raw if channel_att_sum is None else channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
        return x * scale.expand_as(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        mean_pool = torch.mean(x, 1).unsqueeze(1)
        return torch.cat((max_pool, mean_pool), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                 padding=(kernel_size // 2), relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):
    """
    自动通道适配的 CBAM —— 完全兼容 YOLO11 的 width scaling
    """
    def __init__(self, gate_channels=None, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super().__init__()
        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio
        self.pool_types = pool_types
        self.no_spatial = no_spatial

        self.ChannelGate = None
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def _init_channel_gate(self, C):
        if self.ChannelGate is None:
            self.ChannelGate = ChannelGate(
                C,
                reduction_ratio=self.reduction_ratio,
                pool_types=self.pool_types
            ).to(next(self.parameters()).device)

    def forward(self, x):
        C = x.shape[1]
        self._init_channel_gate(C)

        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
```
## 1.2 其他
和[[EMA注意力机制]]一样
# 2. Yaml
## 2.1 Bcakbone
```yaml
# YOLO11 with CBAM (auto-channel version)

nc: 80
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]           # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]          # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]          # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]          # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]         # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]            # 9
  - [-1, 1, CBAM, []]                   # ★ AUTO CBAM (自适应通道)
  - [-1, 2, C2PSA, [1024]]              # 11

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[17, 20, 23], 1, Detect, [nc]]

```
