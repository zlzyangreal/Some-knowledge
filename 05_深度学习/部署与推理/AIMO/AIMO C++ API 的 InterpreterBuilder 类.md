# 1 build_interpretper_from_model_and_config
![[Screenshot 2024-06-09 210056.png]]
# 2.成员函数 build_interpretper_from_path
![[Screenshot 2024-06-16 173556.png]]
# 3.成员函数 build_interpretper_from_model
![[Screenshot 2024-06-16 173659.png]]

## 4. 三种 build 方式的区别

| 成员函数 | 输入 | 适用场景 |
| --- | --- | --- |
| `build_interpretper_from_model_and_config(model, config)` | 内存里的 Model 对象 + Config | 最常用；模型已在内存中 |
| `build_interpretper_from_model(model)` | 只有 Model | 用默认 Config |
| `build_interpretper_from_path(path)` | 模型文件路径 | 最简单，内部帮你建 Model + 默认 Config |

注意官方函数名里是 `interpretper`（拼写如此），写代码时照抄，别自己“纠正”成 `interpreter`。

## 5. 用法示例

```cpp
// 方式一：路径最短
std::unique_ptr<aimo::Interpreter> interp;
aimo::InterpreterBuilder builder;
if (builder.build_interpretper_from_path("model.aimo", &interp) != aimo::kOk) {
    // 处理错误
}

// 方式二：自己控制 Model 与 Config（想设线程数/加速类型时）
aimo::Model model("model.aimo");
aimo::Config config;
config.set_num_threads(4);          // 字段以头文件为准
aimo::InterpreterBuilder b(&model, &config);
std::unique_ptr<aimo::Interpreter> interp2;
if (b(&interp2) != aimo::kOk) { /* 错误处理 */ }
```

## 6. 什么时候需要自定义 Config

- **线程数**：CPU 推理时调 `num_threads`，一般设成物理核数。
- **加速后端**：GPU/NPU 可用时选 `AccelerateType`（见 [[AIMO C++ API 的枚举类]]）。
- **日志级别**：调试阶段打开 DEBUG，量产关掉（见 [[AIMO C++ API 的其他函数]]）。

## 7. 错误处理建议

Builder 失败通常是：路径错、模型格式不匹配（版本/量化方式不对）、目标后端不支持模型里的算子。
建议在失败时打印 `get_library_version()` 与模型路径，方便定位环境问题。

## 相关

- [[AIMO C++ API 的 Interpreter 类]]
- [[AIMO API]] — 整体调用顺序
- [[AIMO使用]]
