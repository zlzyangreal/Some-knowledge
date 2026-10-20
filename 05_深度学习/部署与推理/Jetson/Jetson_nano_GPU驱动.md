# 1.启动自带CUDA
jetson nano默认已经安装了CUDA10.0，但是直接运行 nvcc -V是不会成功的，需要你把CUDA的路径写入环境变量中。
```bash
sudo vim  ~/.bashrc
```
在最后添加
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
然后保存退出
最后 source 一下这个文件。
```bash
source ~/.bashrc
nvcc -V
```
结果
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_28_22:34:44_PST_2021
Cuda compilation tools, release 10.2, V10.2.300
Build cuda_10.2_r440.TC440_70.29663091_0
```