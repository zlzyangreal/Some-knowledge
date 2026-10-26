## 函数的基本语法如下：
```cpp
cv::Mat imread(const String& filename, int flags = cv::IMREAD_COLOR);
```
- `filename`：这是要读取的图像文件的路径。它可以是相对路径或绝对路径。
- `flags`：这是一个可选参数，用于指定读取图像时的模式。默认情况下，它是 `cv::IMREAD_COLOR`，意味着图像将被读取为彩色图像。其他可选的值包括：
    - `cv::IMREAD_GRAYSCALE`：以灰度模式读取图像。
    - `cv::IMREAD_UNCHANGED`：读取图像，包括其Alpha通道（如果存在）。
返回值是 `cv::Mat` 类型的对象，它是一个多功能的矩阵容器，可以存储图像数据。如果图像成功加载，`cv::Mat` 对象将包含图像数据；如果加载失败（例如，文件不存在或路径错误），`cv::Mat` 对象将为空。