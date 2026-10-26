`KalmanFilter` 类的基本用法如下：
# 1.构造函数
```cpp
KalmanFilter(int dp, int mp, int cp = 0);
```
- `dp` 是动态系统的状态维数。
- `mp` 是测量向量的维数。
- `cp` 是控制向量的维数（如果系统没有控制输入，则为0）。
# 2.重要成员变量
- `transitionMatrix`：状态转移矩阵，描述状态随时间的演变。
- `measurementMatrix`：测量矩阵，将状态映射到测量空间。
- `processNoiseCov`：过程噪声协方差矩阵，描述状态预测误差。
- `measurementNoiseCov`：测量噪声协方差矩阵，描述测量误差。
- `errorCovPre`：先验误差协方差矩阵。
- `errorCovPost`：后验误差协方差矩阵。
- `statePre`：先验状态估计。
- `statePost`：后验状态估计。
# 3.方法

- `predict(const Mat &control = Mat())`：预测下一状态，可选地接受一个控制向量。
- `correct(const Mat &measurement)`：根据测量数据校正预测的状态。
# 4.[[卡尔曼滤波器代码实现|代码实现卡尔曼滤波器]]
