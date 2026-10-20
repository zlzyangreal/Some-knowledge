# 1.[[Opencv自带函数]]
# 2.[[OpenCV 类]]

## 3. 一个完整的最小程序

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img = cv::imread("test.jpg", cv::IMREAD_COLOR);
    if (img.empty()) {                       // 一定判断，路径错不会抛异常
        std::cerr << "读图失败" << std::endl;
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    cv::imshow("edges", edges);
    cv::waitKey(0);                          // 0 = 阻塞等待按键
    cv::destroyAllWindows();
    return 0;
}
```

## 4. 编译链接

CMake 版（推荐）：

```cmake
find_package(OpenCV REQUIRED)
add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE ${OpenCV_LIBS})
target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS})
```

命令行版：

```bash
g++ main.cpp -o demo $(pkg-config --cflags --libs opencv4)
```

## 5. Mat 的四个基本属性

| 属性 | 含义 |
| --- | --- |
| `rows` / `cols` | 高、宽 |
| `channels()` | 通道数（灰度 1、彩色 3、带 alpha 4） |
| `depth()` | 元素类型（`CV_8U`、`CV_32F` …），**不含通道数** |
| `type()` | depth + 通道数，如 `CV_8UC3` |

```cpp
cv::Mat a(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));   // 高、宽、类型、初值
cv::Mat b = cv::Mat::zeros(480, 640, CV_8UC1);
cv::Mat c = a.clone();      // 深拷贝（独立内存）
cv::Mat d = a;              // 浅拷贝（共享数据，引用计数）
```

## 6. BGR 与常用坑

1. `imread` 默认读成 **BGR**；`cvtColor(img, out, COLOR_BGR2RGB)` 才变 RGB。
2. `imread` 路径含中文在 Windows 上会失败，用 `cv::imdecode` + `ifstream` 读取。
3. `imshow` 之后必须 `waitKey()`，否则窗口一闪而过/黑屏。
4. `uint8` 直接 `img + 50` 会溢出回绕，用 `cv::add(img, 50, out)` 或先转 `CV_32F`。
5. ROI 赋值是浅拷贝：`img(rect)` 改动会影响原图，要独立就 `.clone()`。

## 相关

- [[OpenCV简介]]、[[OpenCV-C++安装]]、[[OpenCV CMakeLists.txt链接]]
- [[Opencv自带函数]]、[[OpenCV 类]]、[[Mat的详解和例子]]
- [[imread函数]]、[[imshow函数]]、[[waitKey函数]]、[[namedWindow函数]]
