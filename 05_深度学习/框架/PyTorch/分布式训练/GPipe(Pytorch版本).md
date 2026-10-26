库源码:
* https://github.com/KakaoBrain/torchgpipe
*  https://github.com/pytorch/pippy
*  https://github.com/pytorch/PiPPy

## 原理：流水线并行 + 微批

一张卡放不下整个模型时，把**层切成若干段**放到不同卡上，再把一个 batch 切成若干 **micro-batch**，
让不同卡同时处理不同 micro-batch，靠“流水”填满空闲时间。

```
时间 →
卡0: [m1][m2][m3][m4]
卡1:      [m1][m2][m3][m4]
卡2:           [m1][m2][m3][m4]
```

- micro-batch 越多，流水填充/排空（bubble）占比越小，但小 batch 会让 BN 统计不稳。
- GPipe 用**同步梯度累积**：所有 micro-batch 前向完再统一反向，等价于大 batch 训练。

## 三种并行别搞混

| 并行方式 | 切什么 | 代表 |
| --- | --- | --- |
| 数据并行 DDP | 切数据，每个卡有完整模型 | [[DDP分布式]] |
| 张量并行 TP | 切单层内的矩阵 | Megatron-LM |
| 流水线并行 PP | 切层，按阶段分卡 | GPipe、PiPPy |

真实大模型训练通常是 **3D 并行** = 数据 + 张量 + 流水。

## PiPPy 最小用法（PyTorch 官方流水线库）

```python
import torch
from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

# 假设 model 是一串 nn.Sequential，按层名切分
pipe = pipeline(
    module=model,
    mb_args=(example_input,),          # 用于 traced 的样例输入
    split_spec={"layer4": SplitPoint.BEGINNING},
)
schedule = ScheduleGPipe(pipe, chunks=8)   # 8 个 micro-batch
if pipe.is_first_stage():
    schedule.step(example_input)
else:
    schedule.step()
```

启动方式仍是 `torchrun --nproc_per_node=4 train.py`。

## 老库 torchgpipe

```python
from torchgpipe import GPipe
model = nn.Sequential(*layers)
model = GPipe(model, balance=[len(layers)//2]*2, chunks=8, devices=[0, 1])
```

`torchgpipe` 已停止维护，新项目用 `torch.distributed.pipelining`（PiPPy 已并入 PyTorch 主线）。

## 什么时候值得上流水线

- 单卡显存放不下模型（必须切）。
- 卡间互联快（NVLink > PCIe），否则通信开销吃掉收益。
- 有足够大的全局 batch 来切 micro-batch（至少 4~8 个，否则 bubble 太大）。

只是显存不够放 batch → 先试梯度累积、混合精度、`checkpoint`，比流水线简单得多。

## 相关

- [[DDP(显卡交火)]]、[[DDP分布式]]、[[DDP进一步加速]]
- [[GPipe(tensorflow版本)]]
- [[nccl多卡通讯驱动]] — 多卡通信底座
- [[GPU并行训练]]
