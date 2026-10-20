## 链接方式1,直接找
```cmake
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
```
## 链接方式2,在工程中为了方便移植等直接连接到下载库源码位置
```cmake
set(CMAKE_PREFIX_PATH "/home/User/opencv/install/share/OpenCV")
# set(OpenCV_DIR /home/User/opencv/build/)
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
target_link_libraries(MAIN ${OpenCV_LIBS})
```
* 注意链接到的位置