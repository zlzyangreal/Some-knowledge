`cvtColor` 是 OpenCV 库中的一个函数，用于转换图像的颜色空间。颜色空间是描述图像中颜色的一种方式，不同的颜色空间可以提供不同的信息，适用于不同的图像处理任务。
## 函数的基本语法如下：
```cpp
void cvtColor(InputArray src, OutputArray dst, int code, int dstCn = 0);
```
- `src`：输入图像，类型为 `cv::InputArray`，可以是 `cv::Mat` 或其他兼容的图像数据结构。
- `dst`：输出图像，类型为 `cv::OutputArray`，通常也是 `cv::Mat`。
- `code`：一个整数，指定要应用的颜色转换代码。OpenCV 支持多种颜色转换，例如：
    - `cv::COLOR_BGR2RGB`：将图像从 BGR（蓝-绿-红）转换为 RGB（红-绿-蓝）。
    - `cv::COLOR_RGB2BGR`：将图像从 RGB 转换为 BGR。
    - `cv::COLOR_BGR2GRAY`：将图像转换为灰度图像。
    - `cv::COLOR_GRAY2BGR`：将灰度图像转换为 BGR 彩色图像。
- `dstCn`：目标图像的通道数。如果设置为0，OpenCV 会根据 `code` 自动确定通道数。
## 这个函数可以用于多种目的，包括但不限于：
- 将图像从一种颜色空间转换到另一种颜色空间，以匹配特定的图像处理算法或显示需求。
- 将彩色图像转换为灰度图像，以减少计算量或用于特定的图像分析任务。