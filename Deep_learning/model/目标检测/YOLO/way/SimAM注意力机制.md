# 1. Code
ultralytics/ultralytics/nn/modules file_tree
```bash
├── activation.py
├── attention_module
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   └── simam.cpython-310.pyc
│   └── simam.py
├── block.py
├── conv.py
├── head.py
├── __init__.py
├── __pycache__
│   ├── block.cpython-310.pyc
│   ├── conv.cpython-310.pyc
│   ├── head.cpython-310.pyc
│   ├── __init__.cpython-310.pyc
│   ├── transformer.cpython-310.pyc
│   └── utils.cpython-310.pyc
├── transformer.py
└── utils.py
```
simam.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    """SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks."""
    def __init__(self, ch=None, e_lambda=1e-4):  
        super().__init__()
        self.e_lambda = e_lambda
    def forward(self, x):
        # x: [B, C, H, W]
        n = x.shape[2] * x.shape[3] - 1  
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)  # (x - mean)^2
        v = d.sum(dim=[2, 3]) / n + 1e-8  
        E_inv = d / (4 * (v + self.e_lambda).unsqueeze(-1).unsqueeze(-1)) + 0.5  
        return x * torch.sigmoid(E_inv)
```
`/home/alphafox/Desktop/Code/ultralytics/ultralytics/nn/modules/attention_module/__init__.py`
```python
from .simam import SimAM
```
`/home/alphafox/Desktop/Code/ultralytics/ultralytics/nn/modules/__init__.py`
```python
from .block import (
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    CIB,
    DFL,
    ELAN1,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    Attention,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    MaxSigmoidAttnBlock,
    Proto,
    RepC3,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    SCDown,
    TorchVision,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    Index,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import (
    OBB,
    Classify,
    Detect,
    LRPCHead,
    Pose,
    RTDETRDecoder,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

from .attention_module.simam import SimAM #add


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3k2",
    "SCDown",
    "C2fPSA",
    "C2PSA",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "YOLOEDetect",
    "YOLOESegment",
    "v10Detect",
    "LRPCHead",
    "ImagePoolingAttn",
    "MaxSigmoidAttnBlock",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "AConv",
    "ELAN1",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "TorchVision",
    "Index",
    "A2C2f",
    "SimAM",#add
)
```
/home/alphafox/Desktop/Code/ultralytics/ultralytics/nn/tasks.py
add code
```python
from ultralytics.nn.modules import *
from ultralytics.nn.modules.attention_module.simam import SimAM
```
# 2. Yaml
## 2.1 only backbone
```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 181 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
# YOLO11n backbone + SimAM
backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, SimAM, []]         # after downsample
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 1, SimAM, []]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 1, SimAM, []]
  - [-1, 2, C3k2, [512, False, 0.25]]

  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 1, SimAM, []]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 1, SimAM, []]
  - [-1, 2, C3k2, [1024, True]]

  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]]   # 10


# YOLO11n head
# head:
#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 6], 1, Concat, [1]] # cat backbone P4
#   - [-1, 2, C3k2, [512, False]] # 13

#   - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
#   - [[-1, 4], 1, Concat, [1]] # cat backbone P3
#   - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

#   - [-1, 1, Conv, [256, 3, 2]]
#   - [[-1, 13], 1, Concat, [1]] # cat head P4
#   - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

#   - [-1, 1, Conv, [512, 3, 2]]
#   - [[-1, 10], 1, Concat, [1]] # cat head P5
#   - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

#   - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 16
  - [[-1, 9], 1, Concat, [1]]                   # 17  concat backbone P4 (index 9)
  - [-1, 2, C3k2, [512, False]]                 # 18

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 19
  - [[-1, 5], 1, Concat, [1]]                   # 20  concat backbone P3 (index 5)
  - [-1, 2, C3k2, [256, False]]                 # 21  <-- P3 output

  - [-1, 1, Conv, [256, 3, 2]]                  # 22
  - [[-1, 18], 1, Concat, [1]]                  # 23  concat 上一步 (index 18 -> head P4 pre)
  - [-1, 2, C3k2, [512, False]]                 # 24  <-- head P4 output

  - [-1, 1, Conv, [512, 3, 2]]                  # 25
  - [[-1, 15], 1, Concat, [1]]                  # 26  concat backbone P5 (index 15)
  - [-1, 2, C3k2, [1024, True]]                 # 27  <-- head P5 output

  - [[21, 24, 27], 1, Detect, [nc]]             # Detect(P3(idx21), P4(idx24), P5(idx27))

```
## 2.2 Backbone + Head
```yaml
# Ultralytics YOLO11n + SimAM Attention Module 
# SimAM in both backbone and head
 
# Parameters 
nc: 80  # number of classes 
scales: 
  n: [0.50, 0.25, 1024] 
 
# Backbone with SimAM after each downsampling Conv 
backbone: 
  # P1/2 
  - [-1, 1, Conv, [64, 3, 2]]      # 0 
  - [-1, 1, SimAM, []]             # 1 
   
  # P2/4 
  - [-1, 1, Conv, [128, 3, 2]]     # 2 
  - [-1, 1, SimAM, []]             # 3 
  - [-1, 2, C3k2, [256, False, 0.25]]  # 4 
   
  # P3/8 
  - [-1, 1, Conv, [256, 3, 2]]     # 5 
  - [-1, 1, SimAM, []]             # 6 
  - [-1, 2, C3k2, [512, False, 0.25]]  # 7  <-- P3 output 
   
  # P4/16 
  - [-1, 1, Conv, [512, 3, 2]]     # 8 
  - [-1, 1, SimAM, []]             # 9 
  - [-1, 2, C3k2, [512, True]]     # 10 <-- P4 output 
   
  # P5/32 
  - [-1, 1, Conv, [1024, 3, 2]]    # 11 
  - [-1, 1, SimAM, []]             # 12 
  - [-1, 2, C3k2, [1024, True]]    # 13 
   
  - [-1, 1, SPPF, [1024, 5]]       # 14 
  - [-1, 2, C2PSA, [1024]]         # 15 <-- P5 output 
 
# Head - FPN + PAN structure with SimAM
head: 
  # FPN: Top-down pathway 
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 16 
  - [[-1, 10], 1, Concat, [1]]      # 17 (concat with P4 from backbone idx 10) 
  - [-1, 2, C3k2, [512, False]]     # 18 
  - [-1, 1, SimAM, []]              # 19 (SimAM after FPN P4)
   
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 20 
  - [[-1, 7], 1, Concat, [1]]       # 21 (concat with P3 from backbone idx 7) 
  - [-1, 2, C3k2, [256, False]]     # 22 (P3/8-small) 
  - [-1, 1, SimAM, []]              # 23 (SimAM after FPN P3)
   
  # PAN: Bottom-up pathway 
  - [-1, 1, Conv, [256, 3, 2]]      # 24 
  - [[-1, 19], 1, Concat, [1]]      # 25 (concat with head P4 from idx 19)
  - [-1, 2, C3k2, [512, False]]     # 26 (P4/16-medium) 
  - [-1, 1, SimAM, []]              # 27 (SimAM after PAN P4)
   
  - [-1, 1, Conv, [512, 3, 2]]      # 28 
  - [[-1, 15], 1, Concat, [1]]      # 29 (concat with P5 from backbone idx 15) 
  - [-1, 2, C3k2, [1024, True]]     # 30 (P5/32-large) 
  - [-1, 1, SimAM, []]              # 31 (SimAM after PAN P5)
   
  # Detection head 
  - [[23, 27, 31], 1, Detect, [nc]]  # 32 Detect(P3, P4, P5)
```