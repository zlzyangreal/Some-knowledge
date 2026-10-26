树莓派
------
https://www.nxrte.com/jishu/17523.html

## 最快：包管理器

```bash
sudo apt update && sudo apt install -y ffmpeg
ffmpeg -version
```

缺点是版本偏旧，且不带硬件编码（h264_nvenc / h264_v4l2m2m）。

## 需要新特性：源码编译

```bash
sudo apt install -y build-essential nasm yasm pkg-config \
    libx264-dev libx265-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev

git clone https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg
./configure --prefix=/usr/local --enable-gpl --enable-libx264 --enable-libx265 \
            --enable-libvpx --enable-libmp3lame --enable-libopus --enable-shared
make -j$(nproc) && sudo make install && sudo ldconfig
```

树莓派额外加 `--enable-v4l2-m2m --enable-libdrm`，才能用 `h264_v4l2m2m` 硬件编码。

## 验证硬件编码

```bash
ffmpeg -hide_banner -encoders | grep -E "nvenc|v4l2|qsv"
```

## 常用命令速查

```bash
# 查看文件信息
ffprobe input.mp4

# 转码（H.264 + AAC）
ffmpeg -i in.mkv -c:v libx264 -crf 23 -c:a aac -b:a 128k out.mp4

# 抽帧（每 1 秒一张）
ffmpeg -i in.mp4 -vf fps=1 frames/%04d.jpg

# 截取时间段
ffmpeg -ss 00:00:10 -to 00:00:20 -i in.mp4 -c copy clip.mp4

# 推流（配合 nginx-rtmp）
ffmpeg -re -i in.mp4 -c:v libx264 -preset veryfast -f flv rtmp://127.0.0.1/live/test
```

## 相关

- [[nginx与rtmp]]、[[VLC软件本地推拉流]]
- [[gstreamer]]、[[v4l2-ctl]]、[[查看摄像头挂载]]
- 参考：[ffmpeg 编译安装](https://www.nxrte.com/jishu/17523.html)
