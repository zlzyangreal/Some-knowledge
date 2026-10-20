1. **函数原型**：
```cpp
int cv::imshow(const String& winname, InputArray mat);
```
2. **参数**：
- `winname`：要显示图像的窗口的名称。这个名称应该与之前使用 `namedWindow` 函数创建的窗口名称相匹配。
- `mat`：一个 `cv::Mat` 对象，表示要显示的图像。这个对象必须至少有一个通道，且数据类型可以是8位无符号整数（灰度图像）、8位或16位 signed整数、32位浮点数等。