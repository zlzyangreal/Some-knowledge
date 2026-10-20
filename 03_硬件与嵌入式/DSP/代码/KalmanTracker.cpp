#include "KalmanTracker.hpp"

int KalmanTracker::init(float x, float y) {  
	KF.init(4, 2, 0); //卡尔曼的4个动量参数和2个测量参数
	// 测量
	measurement = Mat_<float>::zeros(2, 1);
	measurement.at<float>(0, 0) = x;
	measurement.at<float>(1, 0) = y; // measurement.at<float>(0, 0) = y; 
	// 动量
	KF.statePre.setTo(0);
	KF.statePre.at<float>(0, 0) = x;
	KF.statePre.at<float>(1, 0) = y;
	KF.statePost.setTo(0);   // 设置部分值为指定值
	KF.statePost.at<float>(0, 0) = x;
	KF.statePost.at<float>(1, 0) = y;
 
	setIdentity(KF.transitionMatrix);  //初始化一个缩放矩阵 s：分配给对角线的元素值
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(.005)); // adjust this for faster convergence - but higher noise  快速收敛，噪音较大
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));
	return 0;
}

int KalmanTracker::predict() {
	Mat prediction = KF.predict();  //计算预测状态=>来自kalmanCorrect()函数
	predictPt_ = Point(prediction.at<float>(0), prediction.at<float>(1));
	// predict state
	KF.statePre.copyTo(KF.statePost);
	KF.errorCovPre.copyTo(KF.errorCovPost);
	return 0;
}

int KalmanTracker::correct(float x, float y) {
	measurement(0) = x; measurement(1) = y;
	// udpate measurement, estimate yes or no
	Mat estimated = KF.correct(measurement);
	correctPt_ = Point(estimated.at<float>(0), estimated.at<float>(1));  //更新后的值再传入
	return 0;
}

int KalmanTracker::trace() {
	if (predictPt_.x == 0 && predictPt_.y == 0) {
		cout << "[INFO] predictPt_ is init" << endl;
		goto exit;
	}
	if (tracePt_.size() >= TacePtSize) {
		tracePt_.pop();
		tracePt_.push(predictPt_);
	} else {
		tracePt_.push(predictPt_);
	}
	return 0;
exit:
	return 0;
}

int KalmanTracker::trace_draw(Mat& IO_frame) {
	if (tracePt_.size() == 0) {
		cout << "[INFO] tracePt_ is empty" << endl;
		goto exit;
	}
	{
		Point oldPt = tracePt_.front(); tracePt_.pop(); tracePt_.push(oldPt);
		for (unsigned long i=0; i<tracePt_.size()-1; ++i) {
			Point nowPt = tracePt_.front(); tracePt_.pop(); tracePt_.push(nowPt);
			line(IO_frame, nowPt, oldPt, Scalar(0, 255, 0), 2); // 画线
			oldPt = nowPt; // 更新
		}
		return 0;
	}
    return 0;
exit:
	return 0;
}
