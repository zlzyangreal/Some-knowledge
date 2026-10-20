https://blog.csdn.net/weixin_41805734/article/details/120698714

## 1. 做启动盘

- 工具：Rufus（Windows）或 `dd`（Linux）。
- 镜像：Ubuntu 官网 LTS 版（22.04 / 24.04），服务器用 Server 版，开发机用 Desktop 版。
- 分区表：新主板选 GPT + UEFI；老机器可能是 MBR + Legacy。

```bash
# Linux 下写盘（/dev/sdX 是 U 盘，务必确认，写错会毁硬盘）
sudo dd if=ubuntu-24.04.iso of=/dev/sdX bs=4M status=progress oflag=sync
```

## 2. 安装时怎么分区

| 挂载点 | 建议大小 | 说明 |
| --- | --- | --- |
| `/` | 60–100 GB | 系统和软件 |
| `/home` | 剩余空间 | 个人文件，重装系统可保留 |
| `swap` | 内存大小或 8 GB | 内存 ≥16G 可以只留一个小 swap |
| `/boot/efi` | 512 MB | UEFI 必须，已有 Windows 的不要格式化 |

双系统：先在 Windows 里用磁盘管理压缩出未分配空间，再在 Ubuntu 安装界面选“其他选项”手动挂载，
**不要选“清除整个磁盘”**。装完如果开机直接进 Windows，用 `efibootmgr` 或 BIOS 调整启动顺序。

## 3. 装完立刻做的 6 件事

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl vim htop net-tools openssh-server
sudo timedatectl set-timezone Asia/Shanghai
sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
sudo apt update
sudo systemctl enable --now ssh
```

显卡机器再装驱动（见 [[安装显卡驱动、CUDA以及cuDNN]]），然后 `nvidia-smi` 验证。

## 4. 常见问题

- 安装界面花屏/黑屏：启动项加 `nomodeset`。
- 没有 WiFi：安装时勾选“安装第三方驱动”，或装完用有线网 `apt install bcmwl-kernel-source`。
- 时间差 8 小时：见 [[LInu时区问题]]。

## 相关

- [[linuxubuntu初始配置]]
- [[linux安装依赖关系冲突解决方案]]
- [[添加一块磁盘]]、[[linux系统备份与恢复]]
- 参考：[Ubuntu 安装教程](https://blog.csdn.net/weixin_41805734/article/details/120698714)
