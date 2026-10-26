**！！！切记不要下驱动，下ToolKit**
- [[CUDA驱动破坏补救|下载不同版本CUDA，导致驱动破坏补救]]
![[Snipaste_2025-10-22_23-07-44.png]]
[按照官网指示](https://developer.nvidia.com/cuda-toolkit-archive)
## ⚠️ 若提示 “No such file or directory”（常见错误）
则**手动创建** apt 源文件：
```bash
echo "deb file:/var/cuda-repo-ubuntu2004-11-6-local /" | sudo tee /etc/apt/sources.list.d/cuda-11-6-local.list
```
然后再导入 NVIDIA 公钥`sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub`
# !!!!!!! 安装 Toolkit（仅工具链，不含驱动）

`sudo apt-get install -y cuda-toolkit-11-6`

> 💡 注意：**不要执行 `sudo apt-get install cuda-11-6`**，  
> 否则会强制安装旧驱动覆盖 12.5！


# 7、验证安装

`/usr/local/cuda-11.6/bin/nvcc -V`

输出应类似：

`Cuda compilation tools, release 11.6, V11.6.112`