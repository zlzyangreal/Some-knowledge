1. **函数原型**：
```cpp
int cv::namedWindow(const String& winname, int flags = WINDOW_NORMAL);
```
2. **参数**：
    - `winname`：窗口的名称，用于标识窗口。
    - `flags`（可选）：指定窗口的属性。例如，`WINDOW_NORMAL` 创建一个可调整大小的窗口，而 `WINDOW_AUTOSIZE` 创建一个窗口，其大小自动适应显示的图像。
3. **返回值**：函数返回一个整数，通常这个值是1，表示窗口创建成功。如果窗口已经存在，则返回0。
4. **用途**：`namedWindow` 用于初始化窗口，以便后续可以使用 `imshow` 函数在其中显示图像。
5. **示例用法**：
```cpp
#include <opencv2/opencv.hpp>
int main() {
    // 创建一个窗口，名称为 "My Image"
    cv::namedWindow("My Image", cv::WINDOW_NORMAL);
    // 加载图像并显示
    cv::Mat image = cv::imread("path_to_image.jpg");
    if (!image.empty()) {
        cv::imshow("My Image", image);
        cv::waitKey(0); // 等待用户按键
    } else {
        std::cerr << "Error loading image." << std::endl;
    }
    // 销毁窗口
    cv::destroyWindow("My Image");
    return 0;
}
```