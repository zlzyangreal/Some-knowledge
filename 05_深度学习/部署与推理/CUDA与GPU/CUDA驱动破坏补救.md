2**手打别复制！！！**
- [[helpcuda.txt|复制文件]]
```bash
# ==== STEP 1: 移除旧版本 ====
sudo apt-get remove --purge 'nvidia-*' 'cuda*' 'libcudnn*' -y
sudo apt-get autoremove -y
sudo apt-get clean
sudo rm -rf /usr/local/cuda*
sudo rm -rf /etc/apt/sources.list.d/cuda*
sudo rm -rf /etc/apt/sources.list.d/nvidia*
sudo dpkg --purge nvidia-dkms-510 cuda-drivers-510 nvidia-driver-510 || true
sudo dpkg --configure -a
sudo apt-get -f install -y

# ==== STEP 2: 添加 CUDA 12.5 官方仓库 ====
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# ==== STEP 3: 安装 CUDA 12.5 ====
sudo apt-get install -y cuda-12-5

# ==== STEP 4: 安装推荐驱动（可选但建议）====
sudo apt-get install -y nvidia-driver-550

# ==== STEP 5: 清理缓存并提示重启 ====
sudo apt-get autoremove -y
sudo apt-get clean

```
重启：

```bash
sudo reboot
```

然后执行：

```bash
nvidia-smi
```

应该输出类似：

`Driver Version: 550.xx CUDA Version: 12.5`

再执行：

```bash
/usr/local/cuda/bin/nvcc -V
```

输出：

`Cuda compilation tools, release 12.5, V12.5.x`