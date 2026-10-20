https://blog.csdn.net/wanggao_1990/article/details/81591589
https://blog.csdn.net/weixin_45824920/article/details/115445714
https://blog.csdn.net/zzddada/article/details/116989173
https://willtian.cn/?p=1436
https://blog.csdn.net/qq_16377055/article/details/131604745

## 作用

`nginx-rtmp-module` 把 nginx 变成一个 RTMP 服务器：OBS/ffmpeg 往里推流，多个客户端再从里面拉流。
适合局域网直播、摄像头画面分发、给算法做远程视频源。

## 安装

```bash
sudo apt install -y nginx libnginx-mod-rtmp   # Ubuntu 20.04+ 自带模块包
# 或者源码编译 --add-module=../nginx-rtmp-module
```

## 最小配置（/etc/nginx/nginx.conf 末尾追加）

```nginx
rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;
            # 允许任意 IP 拉流（默认只允许本机）
            allow play all;
        }
    }
}
```

```bash
sudo nginx -t && sudo systemctl restart nginx
```

## 推流与拉流

```bash
# 用 ffmpeg 推本地文件
ffmpeg -re -i test.mp4 -c:v libx264 -preset veryfast -f flv rtmp://127.0.0.1/live/test

# 推摄像头
ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset veryfast -f flv rtmp://127.0.0.1/live/cam

# 拉流（VLC / ffplay）
ffplay rtmp://127.0.0.1/live/test
```

OBS 里：服务器填 `rtmp://IP/live`，串流密钥填 `test`。

## 转 HLS（浏览器可播）

```nginx
application live {
    live on;
    hls on;
    hls_path /tmp/hls;
    hls_fragment 4s;
}
```

再配一个 http location 指向 `/tmp/hls`，浏览器打开 `http://IP/hls/stream.m3u8`。

## 排错

- 推流成功但拉不到：检查 `allow play all;` 和防火墙 1935 端口。
- Nginx 启动失败：`nginx -t` 看配置语法，注意 rtmp 块必须在 `events{}` 之外。
- 延迟高：减小 `chunk_size`、降低 GOP（`-g 30`），或改用 WebRTC/SRT。

## 相关

- [[VLC软件本地推拉流]]
- [[ffmpeg安装]]
- [[rtsp协议]]
- 参考：[nginx-rtmp 搭建](https://blog.csdn.net/wanggao_1990/article/details/81591589)、[RTMP 直播服务配置](https://blog.csdn.net/weixin_45824920/article/details/115445714)
