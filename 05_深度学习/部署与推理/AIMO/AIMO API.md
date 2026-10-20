# 1.[[AIMO C++ API 的枚举类]]
# 2.[[AIMO C++ API 的 Model 类]]
# 3.[[AIMO C++ API 的 Config 类]]
# 4.[[AIMO C++ API 的 Interpreter 类]]
# 5.[[AIMO C++ API 的 InterpreterBuilder 类]]
# 6.[[AIMO C++ API 的其他函数]]

## 整体结构

AIMO 的 C++ 接口是**类 TFLite 风格**的三层结构：

```
Model（模型文件 + 描述）
  └─ InterpreterBuilder（用 Model + Config 造出一个 Interpreter）
        └─ Interpreter（真正的执行器：分配张量、设输入、invoke、取输出）
```

使用顺序固定为：`Model` → `Config` → `InterpreterBuilder.build_*` → `Interpreter.init/load` → `set_input_tensor` → `invoke` → `get_output_tensor`。

## 最小调用骨架

```cpp
#include "aimo/aimo.h"

aimo::Model model("model.aimo");            // 加载模型文件
aimo::Config config;                        // 运行配置（线程数、加速类型等）

aimo::InterpreterBuilder builder(&model, &config);
std::unique_ptr<aimo::Interpreter> interp;
builder(&interp);                            // 构造解释器
interp->init();                              // 初始化运行时
interp->load_model();                        // 载入权重/图

// 输入
interp->set_input_tensor(0, input_data, input_bytes);
interp->invoke();                            // 同步推理
// 输出
interp->get_output_tensor(0, output_data, output_bytes);
```

## 各篇笔记对应关系

| 类/主题 | 笔记 |
| --- | --- |
| 数据类型、实现后端、日志级别等枚举 | [[AIMO C++ API 的枚举类]] |
| 模型文件的封装 | [[AIMO C++ API 的 Model 类]] |
| 运行参数 | [[AIMO C++ API 的 Config 类]] |
| 执行器（init/invoke/张量读写） | [[AIMO C++ API 的 Interpreter 类]] |
| 构造器（三种 build 方式） | [[AIMO C++ API 的 InterpreterBuilder 类]] |
| 版本与日志函数 | [[AIMO C++ API 的其他函数]] |

## 使用要点

1. **输入布局**：端侧推理常见 NCHW/HWC 两种，模型转换时定死，C++ 侧按同样布局喂数据，否则结果全错但不报错。
2. **量化模型**：int8 模型的输入通常是 `uint8` 且带 `zero_point/scale`，别直接喂 float。
3. **张量大小**：`set_input_tensor` 传字节数，不是元素个数。
4. **内存**：输入输出 buffer 由调用方提供时不要提前释放；AIMO 内部不接管所有权。
5. 具体函数签名以 SDK 头文件为准（笔记里的截图来自官方文档）。

## 相关

- [[AIMO使用]]、[[C++部署代码笔记]]
- [[AIMO索引]]（本目录总览）
- [[香橙派环境搭建]]、[[RKNN模型部署]] — 其他端侧方案对照
