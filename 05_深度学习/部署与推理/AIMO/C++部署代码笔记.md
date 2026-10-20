# 1.模型初始化
## 1.1张量定义
```cpp
std::vector<std::vector<uint32_t>> input_shapes = {{1,640,640,3}};
std::vector<std::vector<uint32_t>> output_shapes = {{1,40,40,255}, {1,20,20,255}, {1,80,80,255}};
```
* 1. `std::vector<std::vector<uint32_t>>`: 这是一个嵌套的`vector`，外层是一个`vector`，其元素是内层的`vector`。`uint32_t`是一个无符号的32位整数类型，通常用于表示非负整数。
* 2. `input_shapes`: 这个变量存储了一个输入张量的形状。在这个例子中，它是一个四维数组，其形状为`{1, 640, 640, 3}`，意味着：   
    - 第一个维度是批大小（batch size），这里是1，表示单个图像。
    - 第二和第三个维度是图像的高度和宽度，都是640像素。
    - 第四个维度是颜色通道数，这里是3，表示RGB三通道。
* 3. `output_shapes`: 这个变量存储了多个输出张量的形状。在这个例子中，它包含了三个不同形状的输出张量：
    - 第一个输出张量的形状是`{1, 40, 40, 255}`，表示：
        - 批大小仍然是1。
        - 空间维度被降采样到40x40像素。
        - 通道数变为255，这可能表示该层输出的是类别概率分布，255通常用于表示颜色分类问题中的类别数。
    - 第二个输出张量的形状是`{1, 20, 20, 255}`，空间维度进一步被降采样到20x20像素。
    - 第三个输出张量的形状是`{1, 80, 80, 255}`，空间维度是80x80像素。
这种表示通常用于深度学习模型的输出，特别是在卷积神经网络（CNN）中，其中输入通常是一张图像，而输出可以是多个特征图（feature maps），每个特征图可能对应不同的层级或不同的任务（如分类、检测等）。
## 1.2设置深度学习模型的输入和输出属性，包括它们的形状和数据类型
```cpp
model->set_model_properties(input_shapes, DataType::TYPE_FLOAT32, output_shapes, DataType::TYPE_FLOAT32);
```
* `set_model_properties` 是[[AIMO C++ API 的 Model 类]]成员函数
* `DataType` 结构体
	```cpp
		enum struct DataType : uint8_t{
		    TYPE_DEFAULT = 0,
		    TYPE_UINT8,
		    TYPE_INT8,
		    TYPE_UINT32,
		    TYPE_FLOAT32,
		    TYPE_INT32,
		    TYPE_INT64,
		};
	```
## 1.3
```cpp
Config* config = Config::create_instance();
```
* [[AIMO C++ API 的 Config 类]]
## 1.4 解释器
```cpp
std::unique_ptr<Interpreter>&& fast_interpreter = InterpreterBuilder::build_interpretper_from_model_and_config(model, config);
```
* [[AIMO C++ API 的 Interpreter 类]]
