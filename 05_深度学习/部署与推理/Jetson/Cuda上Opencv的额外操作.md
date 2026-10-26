**OpenCV 的 CUDA 支持需要 `opencv_contrib` 仓库中的 `cudev` 模块**
- 手动下载
```bash
		https://github.com/opencv/opencv_contrib/releases/tag/4.8.0
```

```bash
git clone -b 4.5.0 https://github.com/opencv/opencv_contrib.git
```
- 4.5.0是版本
配置插件在软件包内
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="5.3" \  # Jetson Nano的架构
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUBLAS=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \  # 指向你的opencv_contrib/modules目录
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF ..
```
- `CUDA_ARCH_BIN`参数在 `jtop`的信息栏 `INFO`里面的 `CUDA Arch BIN`，x86就直接在`nvidia-smi`查询
- [[指令.txt|输入指令]]
打开python还需要加入
```shell
      -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
      -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
      -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
```