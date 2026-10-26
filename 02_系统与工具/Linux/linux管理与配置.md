# [[linux定时任务调度]]
# [[linux进程管理]]
# [[linux日志管理]]
# [[linux网络配置]]
# [[linuxubuntu初始配置]]

## 系统管理四条主线

1. **进程与服务**：`ps aux` 看全量进程、`top/htop` 看实时占用、`systemctl` 管开机自启服务，见 [[linux进程管理]]。
2. **日志**：`journalctl -u 服务名 -f` 追服务日志，`/var/log/` 下按应用分文件，见 [[linux日志管理]]。
3. **定时任务**：`crontab -e` 写用户级任务，`/etc/cron.d/` 放系统级，见 [[linux定时任务调度]]。
4. **网络**：`ip a` 看网卡、`nmcli`/`netplan` 配地址、`ss -lntp` 看监听端口，见 [[linux网络配置]]。

## 上手顺序（新机器装完系统后）

```bash
# 1. 更新与基础工具
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl vim htop net-tools

# 2. 主机名与时区
sudo hostnamectl set-hostname dev
sudo timedatectl set-timezone Asia/Shanghai      # 见 LInu时区问题

# 3. 换国内源（Ubuntu）
sudo sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 4. 开 SSH
sudo apt install -y openssh-server
sudo systemctl enable --now ssh

# 5. 常用目录权限
sudo chown -R $USER:$USER ~/work
```

## 排查三板斧

- 服务起不来：`systemctl status xxx` → `journalctl -u xxx -n 100 --no-pager`
- 磁盘满：`df -h` 找到分区 → `du -sh /* 2>/dev/null | sort -h` 逐层定位
- 端口被占：`ss -lntp | grep 端口` → `kill -9 PID`

## 相关

- [[Linux指令操作]]、[[Linux基础理论索引]]、[[系统管理索引|Linux 系统管理]]
- [[linuxubuntu初始配置]]、[[添加一块磁盘]]、[[linux找回root密码]]
- [[定制linux系统]]、[[linux系统备份与恢复]]
