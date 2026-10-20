# 1. Code
包括单独的CA，以及将CA融入C3k2
```python
import torch
import torch.nn as nn

# 必须导入 C3k2
from ultralytics.nn.modules.block import C3k2

class h_sigmoid(nn.Module):
    """Hard sigmoid activation function"""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """Hard swish activation function"""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 确保通道数为整数
        channels = int(channels)
        mip = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w

class CA(nn.Module):
    def __init__(self, c1, reduction=32):
        super(CA, self).__init__()
        self.ca = CoordAtt(c1, reduction)
        
    def forward(self, x):
        return self.ca(x)

class C3k2_CA(C3k2):
    """
    C3k2 block with integrated Coordinate Attention (CA).
    使用 *args 透传参数，并强制类型转换，防止参数错位。
    """
    def __init__(self, *args, **kwargs):
        # 参数清洗：确保前三个位置参数是整数
        # args 顺序通常是: c1, c2, n, c3k, e, g, shortcut
        safe_args = list(args)
        
        # 强制转换 c1, c2, n 为 int，解决 TypeError: empty() 报错
        if len(safe_args) >= 1: safe_args[0] = int(safe_args[0]) # c1
        if len(safe_args) >= 2: safe_args[1] = int(safe_args[1]) # c2
        if len(safe_args) >= 3: safe_args[2] = int(safe_args[2]) # n

        # 调用父类 C3k2
        super().__init__(*safe_args, **kwargs)
        
        # 获取输出通道数 c2 用于初始化 CA
        if len(safe_args) >= 2:
            c2 = safe_args[1]
        else:
            c2 = kwargs.get('c2')
            
        # 可以在这里打印一下，确保 c2 不是 0
        if c2 == 0:
            raise ValueError("C3k2_CA 初始化失败：输出通道数 c2 为 0。请检查 tasks.py 是否正确注册了 C3k2_CA 到 base_modules。")

        self.ca = CA(c2)

    def forward(self, x):
        # 先经过 C3k2 提取特征
        x = super().forward(x)
        # 再经过 CA 增强特征
        return self.ca(x)
```
除了和常规的一样加入init和头文件以外，还需要在tasks.py中加入
```python
base_modules = frozenset(
        {
            Classify,
            Conv,
            # ... 省略中间代码 ...
            C2fCIB,
            A2C2f,
            C3k2_CA,  # <--- 【必须添加】在这里加入 C3k2_CA
        }
    )
repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            # ... 省略中间代码 ...
            C2PSA,
            A2C2f,
            C3k2_CA,  # <--- 【必须添加】在这里加入 C3k2_CA
        }
    )
```
# 2. yaml
将CA融入C3k2
```yaml
# Ultralytics YOLO11n with C3k2_CA
# Replaces all standard C3k2 blocks with Attention-Integrated C3k2 blocks

# Parameters
nc: 80 # number of classes
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  
  # Replace C3k2 with C3k2_CA
  - [-1, 2, C3k2_CA, [256, False, 0.25]] # 2
  
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  
  # Replace C3k2 with C3k2_CA
  - [-1, 2, C3k2_CA, [512, False, 0.25]] # 4
  
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  
  # Replace C3k2 with C3k2_CA
  - [-1, 2, C3k2_CA, [512, True]]        # 6
  
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  
  # Replace C3k2 with C3k2_CA
  - [-1, 2, C3k2_CA, [1024, True]]       # 8
  
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 2, C2PSA, [1024]]             # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]          # cat backbone P4
  
  # Replace C3k2 with C3k2_CA (Neck processing)
  - [-1, 2, C3k2_CA, [512, False]]       # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]          # cat backbone P3
  
  # Replace C3k2 with C3k2_CA (Neck processing - Small scale)
  - [-1, 2, C3k2_CA, [256, False]]       # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]         # cat head P4
  
  # Replace C3k2 with C3k2_CA (Neck processing - Medium scale)
  - [-1, 2, C3k2_CA, [512, False]]       # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]         # cat head P5
  
  # Replace C3k2 with C3k2_CA (Neck processing - Large scale)
  - [-1, 2, C3k2_CA, [1024, True]]       # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]]    # Detect(P3, P4, P5)
```