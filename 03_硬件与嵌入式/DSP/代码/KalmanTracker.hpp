#ifndef _KALMAN_TRACKER_HPP_
#define _KALMAN_TRACKER_HPP_
#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

#define TacePtSize 50 

class KalmanTracker {
public:
    Point predictPt_; 
    Point correctPt_; 
    queue<Point> tracePt_;
    
    KalmanTracker() {}
    ~KalmanTracker() {}
    int init(float x = 0.0f, float y = 0.0f);
    int predict();
    int correct(float x, float y);
    int trace();
    int trace_draw(Mat& IO_frame);
    int trace_clear() {
        while (!tracePt_.empty()) {
            tracePt_.pop();
        }
        return 0;
    }
private:
    KalmanFilter KF;   //卡尔曼滤波器
    // 以下两种类的初始化方法都可以，编译器不会造成歧义
    // 2 measurement variables, 1 state variable
    Mat_<float> measurement { Mat_<float>(2, 1) };   //模板矩阵
    // 4 state variables, 2 measurement variables
    Mat_<float> state = Mat_<float>(4, 1); // (x, y, Vx, Vy)
};

#endif