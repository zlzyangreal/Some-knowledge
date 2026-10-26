# [[关机重启]]
# [[用户管理]]
# [[常用命令索引|帮助指令]]
# [[文件目录]](包含查看部分)
# [[时间日期类]]
# [[搜索查找类]]
# [[压缩解压类]]
# [[磁盘操作]]
# [[文件目录权限]]

## 命令速查（按用途）

| 用途 | 命令 |
| --- | --- |
| 看目录 | `ls -alh`、`tree -L 2`、`pwd` |
| 切换 | `cd -`（回上一个目录）、`cd ~` |
| 看文件 | `cat`、`less`（可搜索 `/关键字`）、`head -n 20`、`tail -f` |
| 找文件 | `find . -name "*.py"`、`locate`、`which` |
| 找内容 | `grep -rn "关键字" .`、`rg`（更快） |
| 统计 | `wc -l`、`du -sh *`、`df -h` |
| 权限 | `chmod 755`、`chown -R user:group` |
| 进程 | `ps aux`、`top`、`htop`、`kill -9 PID` |
| 网络 | `ip a`、`ss -lntp`、`ping`、`curl -I` |
| 压缩 | `tar -czf a.tar.gz dir`、`tar -xzf a.tar.gz` |
| 包管理 | `apt install`、`apt remove`、`dpkg -l` |
| 服务 | `systemctl status/start/enable xxx` |

## 使用习惯

1. 任何命令不熟，先 `man xxx` 或 `xxx --help`，见 [[帮助指令]]。
2. 危险命令先 `echo` 试一遍再执行（`rm -rf` 前先 `ls` 同样的路径）。
3. `Tab` 补全、`Ctrl+R` 搜历史命令、`Ctrl+C` 中断、`Ctrl+Z` 挂起 + `fg` 恢复。
4. 长任务用 `nohup cmd > out.log 2>&1 &` 或 `tmux`，别让终端一断就没了。

## 按主题深入

- 文件与目录：[[文件目录]]、[[文件目录权限]]
- 系统管理：[[linux进程管理]]、[[linux日志管理]]、[[linux定时任务调度]]、[[linux网络配置]]
- 文本处理：[[文本less查看]]、[[文本more查看]]、[[vi与vim]]
- 运维排查：[[磁盘操作]]、[[用户管理]]、[[时间日期类]]

## 相关

- [[linux管理与配置]]
- [[整套cpp工作链]]
- [[linux环境变量]]
