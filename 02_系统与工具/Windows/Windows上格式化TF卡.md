https://digitalixy.com/howto/1036992.html

## 方法一：图形界面（最简单）

1. `Win+X` → 磁盘管理。
2. 找到 TF 卡对应的磁盘（看容量，别选成硬盘）。
3. 右键分区 → 删除卷 → 新建简单卷 → 选 FAT32 / exFAT → 格式化。

## 方法二：diskpart（容量显示异常、分区表损坏时用）

```cmd
diskpart
list disk                 :: 看清楚是磁盘几！按容量认，别按序号猜
select disk 2             :: 假设 TF 卡是磁盘 2
clean                     :: 清空分区表（数据全没）
create partition primary
format fs=fat32 quick     :: 大于 32G 建议 fs=exfat
assign                    :: 分配盘符
exit
```

> [!WARNING]
> `clean` 会立刻抹掉所选磁盘的分区表。执行前务必用 `list disk` 核对容量；
> 拔掉其它 U 盘/移动硬盘再操作更安全。

## 选哪种文件系统

| 文件系统 | 单文件上限 | 适用 |
| --- | --- | --- |
| FAT32 | 4 GB | 单片机 SD 卡、相机、几乎所有设备都认 |
| exFAT | 无实际限制 | 大容量卡、需要在 Windows/Mac 通吃 |
| NTFS | 无 | 只在 Windows 用，很多嵌入式设备不认 |

树莓派/Jetson 刷系统后卡里会有多个分区（boot 是 FAT32，root 是 ext4），
Windows 只能看到 boot 分区，这是正常的，不要用 Windows 去“格式化”整张卡。

## TF 卡写不进去 / 容量变小

- 用了 `dd` 或烧录工具后容量“变小”：是分区表被镜像覆盖，用 `diskpart clean` 或 SD Card Formatter 恢复。
- 卡变只读：多数是寿命到期的自我保护，换卡。
- 烧录 Linux 镜像推荐 balenaEtcher，比 Win32DiskImager 更稳。

## 相关

- [[添加一块磁盘]]
- [[安装JETPACK套件]]、[[树莓派安装Python3.9]]
- 参考：[格式化 TF 卡教程](https://digitalixy.com/howto/1036992.html)
