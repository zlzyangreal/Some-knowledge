# 1.安装显卡驱动
## 1.1 禁用Nouveau
这个为**文件可能不存在**，不存在新建
```bash
vim /etc/modprobe.d/blacklist-nouveau.conf
```
写入
```swift
blacklist nouveau
options nouveau modeset=0
```
更新使其生效

```bash
update-initramfs -u
```

重启机器

```bash
reboot
```

检查是否禁用成功

```bash
lspci | grep nouveau
```

没有内容输出，说明禁用成功。
## 1.2安装gcc/cmake
- [[整套cpp工作链]]
## 1.3安装NVIDIA驱动
### 1.3.1 查看显卡型号
```bash
lspci | grep -i nvidia
```
输出eg
```swift
01:00.0 VGA compatible controller: NVIDIA Corporation Device 2705 (rev a1)
01:00.1 Audio device: NVIDIA Corporation Device 22bb (rev a1)
```
在[网址](https://admin.pci-ids.ucw.cz/read/PC/10de/2208)查询
### 1.3.2 安装NVIDIA驱动
**记住下载驱动文件位置**
Ctrl+Alt+F3 进入 tty 终端模式
输入下面命令关闭显示管理：

```bash
service lightdm stop
```
也可能不是这显示服务
安装（tty下）
```bash
cd /home/username/Downloads  # 此处的目录为你下载的驱动文件所在目录
sh （驱动文件） --no-opengl-files
```
这个指令有输出就安装好了
```bash
nvidia-wsmi
```
# 2 下载CUDA Toolkit
[查询文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
[官网下载](https://developer.nvidia.com/cuda-toolkit-archive)
## 2.1 按照官网给定指令下载
- [[cuda不同版本下载]]
- [[cuda不同版本切换]]
查看系统路径
```swift
cuda/
cuda-12/
cuda-12.5/
```
## 2.2 **查看指定版本**
```bash
ls -l /usr/local/cuda
```
- 输出可能是
	```swift
		/usr/local/cuda -> /etc/alternatives/cuda
	```
追踪实际文件夹
```bash
ls -l /etc/alternatives/cuda
```
## 2.3 链接
```bash
sudo ln -s /usr/local/cuda-12.3 /usr/local/cuda
```
* 以12.3版本为例
# 3 cuDNN下载
[下载网址](https://developer.nvidia.com/rdp/cudnn-archive)
## 3.1 安装
解压，**可能下载到xz文件**
```bash
tar -xvf cudnn-linux-x86_64-9.14.0.64_cuda12-archive.tar.xz
```
### 3.2 移动文件
```bash
sudo cp include/cudnn*.h /usr/local/cuda-12.5/include/
sudo cp lib/libcudnn* /usr/local/cuda-12.5/lib64/
sudo chmod a+r /usr/local/cuda-12.5/include/cudnn*.h /usr/local/cuda-12.5/lib64/libcudnn*
```
### 3.3 验证
```bash
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```