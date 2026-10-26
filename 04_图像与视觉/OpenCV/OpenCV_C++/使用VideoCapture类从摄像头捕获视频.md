```cpp
#include <opencv2/opencv.hpp>
int main() {
    // 打开摄像头
    cv::VideoCapture capture(0); // 0是默认摄像头的设备索引
    if (!capture.isOpened()) {
        // 如果无法打开摄像头，则返回错误
        return -1;
    }
    cv::Mat frame;
    while (true) {
        // 从摄像头读取一帧
        capture.read(frame);
        // 如果正确读取帧，break循环
        if (frame.empty()) {
            break;
        }
        // 显示帧
        cv::imshow("Frame", frame);
        // 按下'q'键退出循环
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }
    // 释放资源
    capture.release();
    return 0;
}
```