# 1.启动虚拟机功能
在电脑上打开“控制面板”->“程序”-> “启动或关闭Windows功能”。
![[docker.png]]
- 按 `Ctrl + Shift + Esc` 打开 **任务管理器**。
- 切换到 **性能** 选项卡，点击左侧的 **CPU**。
- 在右下方查看 **“虚拟化”** 这一项的状态。
# 2. 进入Docker官网

首先先到Docker官网下载最新官方Docker for Windows链接：[Docker下载](https://docs.docker.com/desktop/install/windows-install/)
# 3. 更新一下wsl版本
```bash
wsl --update
```