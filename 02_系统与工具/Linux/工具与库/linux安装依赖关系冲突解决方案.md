# 安装关系冲突解决方案
https://blog.csdn.net/quantum7/article/details/103977710

## 按顺序试这几招

### 1. 先让 apt 自愈

```bash
sudo apt --fix-broken install
sudo dpkg --configure -a
sudo apt update && sudo apt upgrade -y
```

90% 的“依赖冲突”都是上次安装被中断留下的半成品，这两条能修好。

### 2. 看清是谁在打架

```bash
apt-cache policy 包名          # 有哪些版本、从哪个源来
apt-cache depends 包名         # 它依赖谁
dpkg -l | grep 包名            # 当前装的是哪个版本
```

### 3. 卸掉冲突的一方

```bash
sudo apt remove --purge 冲突包
sudo apt autoremove
```

注意：`--purge` 会连配置一起删，别对系统关键包用。

### 4. 锁版本（要共存时）

```bash
sudo apt-mark hold 包名        # 禁止升级
sudo apt-mark unhold 包名
```

典型场景：CUDA 和显卡驱动互相要求特定版本，装好后 `hold` 驱动。

### 5. 处理“混装 dpkg -i 的 .deb”

```bash
# dpkg -i 不会自动装依赖，缺依赖时：
sudo apt install -f            # 让 apt 补齐依赖
```

### 6. 换源解决“找不到版本”

```bash
# Ubuntu 换清华源后 update，很多"版本不存在"就消失了
sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list
```

### 7. 彻底隔离：不要动系统 Python

系统级 Python 包冲突（pip 与 apt 打架）最好的解法是别在系统里装：

```bash
python3 -m venv ~/venvs/py310 && source ~/venvs/py310/bin/activate
# 或 conda create -n py310 python=3.10
```

## 常见坑

- `apt upgrade` 时看到要删除一堆包 → 先按 `n` 取消，确认是不是源混乱。
- 同时启用了多个第三方源（如 docker、nvidia、deadsnakes）→ 版本会互相顶，尽量保持官方源为主。
- 强制 `--force-all` 安装会把系统弄坏，能不用就不用。

## 相关

- [[ubuntu安装]]
- [[Anaconda索引]]、[[anaconda换源]]
- [[安装显卡驱动、CUDA以及cuDNN]]
- 参考：[apt 依赖关系冲突解决](https://blog.csdn.net/quantum7/article/details/103977710)
