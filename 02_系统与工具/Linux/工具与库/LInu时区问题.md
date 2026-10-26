https://blog.csdn.net/yuchangyuan5237/article/details/131428812

## 先看当前状态

```bash
timedatectl                 # 看 Time zone 与是否 NTP 同步
date -R                     # 看带时区的完整时间
cat /etc/timezone           # Debian/Ubuntu 的时区文件
ls -l /etc/localtime        # 指向 /usr/share/zoneinfo/...
```

## 改成东八区（三种等价方式）

```bash
sudo timedatectl set-timezone Asia/Shanghai
# 或
sudo ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# 或临时
export TZ=Asia/Shanghai
```

## 与 NTP 同步

```bash
sudo timedatectl set-ntp true
sudo systemctl restart systemd-timesyncd
timedatectl show-timesync --all
```

## 程序里时间差 8 小时怎么办

| 场景 | 处理 |
| --- | --- |
| C/C++ `localtime` 差了 8 小时 | 容器里没 `TZ`，`docker run -e TZ=Asia/Shanghai` |
| Python `datetime.now()` 是 UTC | 用 `datetime.now(ZoneInfo("Asia/Shanghai"))`，别用 `utcnow()` |
| 数据库存的时间错 | 服务与数据库都要统一时区，最好全存 UTC，展示层再转 |
| 嵌入式板子重启后时间回到 1970 | 没电池/RTC，用 `ntpdate` 或 `hwclock -w` 写回 RTC |

## Docker 容器时区

```dockerfile
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
```

## 相关

- [[linux网络配置]]
- [[linux定时任务调度]]（cron 的时间就是系统时区）
- [[ubuntu安装]]
- 参考：[Linux 时区问题](https://blog.csdn.net/yuchangyuan5237/article/details/131428812)
