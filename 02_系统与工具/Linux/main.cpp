#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
 
int main(int argc, char *argv[])
{
// 打开默认摄像头（设备索引为0）
    cv::VideoCapture cap("/dev/video0");
    
    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera /dev/video1" << std::endl;
        cap.open("/dev/video1");
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera /dev/video1" << std::endl;
            return -1;
        }
    }

    // 创建一个窗口
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat frame;
        
        // 从摄像头读取一帧
        cap >> frame;
        
        // 检查帧是否为空
        if (frame.empty()) {
            std::cerr << "Error: Could not capture frame" << std::endl;
            break;
        }

        // 在窗口中显示帧
        cv::imshow("Camera", frame);

        // 等待1毫秒，检测是否按下了ESC键
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 释放摄像头
    cap.release();
    // 销毁窗口
    cv::destroyAllWindows();
    return 0;
}