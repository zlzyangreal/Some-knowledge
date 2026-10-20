https://cloud.tencent.com/developer/article/2147461

## 一句话原理

VLC 既能当**推流端**（把本地文件/摄像头编码后发出去），也能当**拉流端**（接收并播放）。
协议选 RTSP（低延迟、需服务器）或 UDP/HTTP（临时测试最省事）。

## 最快验证：UDP 直推直收（同一台机器）

```bash
# 推（把 test.mp4 推成 udp 流）
vlc test.mp4 --sout "#standard{access=udp,mux=ts,dst=239.255.255.250:1234}"

# 拉
vlc udp://@239.255.255.250:1234
```

## 摄像头推流（Linux）

```bash
# 先确认设备名，见 查看摄像头挂载 / v4l2-ctl
vlc v4l2:///dev/video0:width=1280:height=720:fps=30 \
    --sout "#transcode{vcodec=h264,vb=2000}:standard{access=udp,mux=ts,dst=239.0.0.1:1234}"
```

## RTSP / RTMP 推流（需要服务器）

```bash
# 推到 RTMP 服务器（服务器搭建见 nginx与rtmp）
vlc test.mp4 --sout "#transcode{vcodec=h264}:rtp{sdp=rtsp://:8554/stream}"

# 拉 RTSP
vlc rtsp://127.0.0.1:8554/stream
```

## 图形界面做法

媒体 → 串流（Stream）→ 添加文件 → 选择“UDP”或“RTP/RTSP” → 填目标地址 → 播放。
接收方：媒体 → 打开网络串流 → 粘贴地址。

## 常见问题

- 拉流卡顿：降低码率/分辨率，或换 UDP 组播（局域网内）。
- 有画面没声音：少了 `acodec`，转码参数里补 `acodec=mp3,ab=128`。
- 防火墙：UDP 端口没放行，`sudo ufw allow 1234/udp`。

## 相关

- [[nginx与rtmp]] — 搭 RTMP 服务器
- [[ffmpeg安装]] — 命令行替代方案，脚本化更方便
- [[rtsp协议]]、[[查看摄像头挂载]]、[[v4l2-ctl]]
- 参考：[VLC 推拉流实践](https://cloud.tencent.com/developer/article/2147461)
