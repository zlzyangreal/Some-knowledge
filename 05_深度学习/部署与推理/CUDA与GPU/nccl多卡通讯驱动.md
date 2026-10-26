[网址](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)

## NCCL 是什么

NVIDIA 的**多卡/多机集合通信库**（类似 MPI 的集合通信，但针对 GPU 优化）。
PyTorch 的 `torch.distributed` 在 GPU 上默认就用 NCCL 作为后端：`init_process_group(backend="nccl")`。
常见集合操作：`all_reduce`（梯度求和）、`all_gather`、`broadcast`、`reduce_scatter`。

## 安装

```bash
# 随 CUDA 一起装（多数情况已包含）
sudo apt install -y libnccl2 libnccl-dev

# 或按官方仓库装指定版本（要和 CUDA 版本匹配）
# https://developer.nvidia.com/nccl/nccl-download
```

验证：

```bash
python -c "import torch; print(torch.cuda.nccl.version())"
ldconfig -p | grep nccl
```

## 典型使用（DDP 已经封装好）

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

# 之后正常 loss.backward() / optimizer.step()，梯度同步由 NCCL 完成
```

启动：

```bash
torchrun --nproc_per_node=4 --nnodes=1 train.py
```

## 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `NCCL error: unhandled system error` | 多卡 P2P 不通（PCIe 拓扑/IOMMU），试 `NCCL_P2P_DISABLE=1` |
| 卡在 init_process_group | 端口被占或防火墙；换 `MASTER_PORT`，同机检查 `localhost` 解析 |
| 卡间速度慢 | 没走 NVLink/PCIe 直连，`nvidia-smi topo -m` 看拓扑 |
| 版本不匹配 | NCCL 版本与 PyTorch 编译时用的不一致，重装匹配版本 |
| 单卡能跑多卡报错 | 常是 `set_device` 漏了，或 DDP 里 `device_ids` 没给 |

调试环境变量：`NCCL_DEBUG=INFO`（日志）、`NCCL_IB_DISABLE=1`（禁用 InfiniBand）、`NCCL_SOCKET_IFNAME=eth0`（指定网卡）。

## 相关

- [[GPU并行训练]]
- [[DDP分布式]]、[[DDP进一步加速]]
- [[安装显卡驱动、CUDA以及cuDNN]]、[[cuda不同版本切换]]
- [NCCL 官方安装指南](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)
