# OpenCV_C++

C++ 版 OpenCV：Mat、Scalar、常用函数与视频读写。

上级：[[OpenCV索引]]　|　总索引：[[总索引]]

## 笔记（20）

- [[KalmanFilter类]] — KalmanFilter 类的基本用法如下：
- [[Mat的详解和例子]] — 数据类型支持：Mat 类可以存储多种数据类型，包括8位无符号整数（用于灰度图像）、32位浮点数（用于深度图像）、BGR/
- [[Mat类convertTo函数]] — convertTo 是 OpenCV 库中 cv::Mat 类的一个成员函数，用于将矩阵转换为不同的数据类型。这个函数非
- [[Mat类zeros函数]] — cv::Mat::zeros 是一个静态成员函数，它用于创建一个初始化为零的矩阵。这个函数非常有用，因为它允许你快速生成
- [[Mat类函数clone()]] — 返回值：返回一个新的 Mat 对象，它是原对象的深拷贝。
- [[Mat类函数empty()]] — 检查 Mat 对象是否没有分配数据或者它的数据矩阵是否为空。如果 Mat 对象为空，empty() 函数返回 true；
- [[OpenCV CMakeLists.txt链接]] — 注意链接到的位置
- [[OpenCV 库使用]] — CMake 版（推荐）：
- [[OpenCV 类]] — 用于捕获视频数据的类。这个类提供了从各种视频源（如文件、摄像头、网络摄像头等）捕获视频帧的功能
- [[OpenCV-C++安装]] — 编译前把conda先关了防止冲突
- [[Opencv自带函数]] — 它用于等待一个键盘事件
- [[Scalar类使用例子]] — 当你使用 cv::Scalar 来创建颜色时，你可以指定每个通道的值（例如，BGR或RGB）。
- [[cvtColor函数]] — cvtColor 是 OpenCV 库中的一个函数，用于转换图像的颜色空间。颜色空间是描述图像中颜色的一种方式，不同的颜
- [[divide函数]] — 在 OpenCV 中，divide 函数用于执行两个矩阵的元素级除法操作。这个函数可以处理不同大小和类型的矩阵，并且支持
- [[imread函数]] — filename：这是要读取的图像文件的路径。它可以是相对路径或绝对路径。
- [[imshow函数]] — winname：要显示图像的窗口的名称。这个名称应该与之前使用 namedWindow 函数创建的窗口名称相匹配。
- [[namedWindow函数]] — winname：窗口的名称，用于标识窗口。
- [[rectangle函数]] — rectangle函数用于在图像上绘制矩形
- [[waitKey函数]] — 功能：waitKey 会阻塞程序的执行，直到用户按下一个键，或者超过指定的等待时间（毫秒）。
- [[使用VideoCapture类从摄像头捕获视频]] （纯代码）
