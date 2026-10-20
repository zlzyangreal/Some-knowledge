# 1.enum DataType
![[Screenshot 2024-06-16 163605.png]]
# 2.enum ImplementType
![[Screenshot 2024-06-16 170444.png]]
* [[IPC是什么]]
# 3.enum FrameworkType
![[Screenshot 2024-06-16 170919.png]]
# 4.enum AccelerateType
![[Screenshot 2024-06-16 171043.png]]
# 5.enum LogLevel
![[Screenshot 2024-06-16 171137.png]]

## 6. 每个枚举在解决什么问题

| 枚举 | 用途 | 典型取值含义 |
| --- | --- | --- |
| `DataType` | 张量元素类型 | FLOAT32 / FLOAT16 / INT8 / UINT8（决定你喂数据时用 float 还是 int8 buffer） |
| `ImplementType` | 用哪套算子实现 | 参考实现 / 优化实现（不同后端） |
| `FrameworkType` | 模型来源框架 | 用于识别模型是 ONNX / TFLite / 自家格式导出 |
| `AccelerateType` | 硬件加速后端 | CPU / GPU / NPU（设备不支持时需回退 CPU） |
| `LogLevel` | 日志级别 | Debug / Info / Warn / Error / None |

## 7. 实用对应关系

```cpp
// 你手里的模型是什么量化方式，决定 set_input_tensor 传什么类型
//   FLOAT32 模型 → std::vector<float>
//   INT8/UINT8 模型 → std::vector<int8_t> / <uint8_t>，且输入要按 scale/zero_point 量化

// 设备没有 NPU 时选 CPU，别硬选导致 build 失败
config.set_accelerate_type(aimo::AccelerateType::kCPU);
```

## 8. 常见坑

1. **int8 模型喂 float**：不会报错，但结果完全乱——因为字节解释方式变了。
2. **AccelerateType 与算子支持**：NPU 后端通常只支持部分算子，遇到不支持的会自动回退或直接失败。
3. **FrameworkType 选错**：模型能加载但推理结果不对，属于“静默错误”，务必用一组已知输入输出做对齐测试。

## 相关

- [[IPC是什么]] — 跨进程通信，端侧多进程部署时会用到
- [[AIMO API]]、[[AIMO C++ API 的 Interpreter 类]]
- [[rknn推理YOLOv8]] — 另一套端侧推理的枚举/配置对照
