# 1.`VideoCapture`
用于捕获视频数据的类。这个类提供了从各种视频源（如文件、摄像头、网络摄像头等）捕获视频帧的功能
eg:[[使用VideoCapture类从摄像头捕获视频]]
# 2.Mat
`Mat` 是一个非常核心的类，代表一个多用途的矩阵（Matrix）。这个类被广泛用于存储和操作图像帧、视频帧、点云数据、深度图等
* [[Mat的详解和例子]]
## 类函数
[empty()](Mat类函数empty().md)：用于检查一个 `Mat` 对象是否为空
[clone()](Mat类函数clone().md): 用于创建一个与原矩阵 `Mat` 对象完全相同的[[拷贝构造函数|深拷贝（deep copy）]]
[[Mat类zeros函数|zeros()]]:用于创建一个初始化为零的矩阵
[[Mat类convertTo函数|convertTo()]]
# 3.Point
`Point` 是一个模板类，用于表示二维空间中的一个点。
## 常用类型
- `Point2i`：用于存储以整数表示的坐标（通常用于像素坐标）。
- `Point2f`：用于存储以浮点数表示的坐标（通常用于亚像素精度的坐标）。
# 4.Scalar
表示4个元素的向量，这些元素可以是任何基本数据类型，如 `int`、`float` 或 `double`。它常用于表示颜色值、归一化因子、图像的像素值等.
* [[Scalar类使用例子|使用例子]]
# 5.Size
用于表示二维尺寸，通常用于图像处理中指定图像的宽度和高度
# 6.KalmanFilter
OpenCV中的 `KalmanFilter` 类是一个实现[[卡尔曼滤波器]]的类。
[[KalmanFilter类|基本使用方法]]