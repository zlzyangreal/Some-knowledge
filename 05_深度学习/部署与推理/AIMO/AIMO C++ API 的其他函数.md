# 1.get_library_version()
![[Screenshot 2024-06-16 173833.png]]
# 2.set_log_level()
![[Screenshot 2024-06-16 173925.png]]
# 3.log_to_stderr()
![[Screenshot 2024-06-16 174038.png]]
# 4.log_to_file()
![[Screenshot 2024-06-16 174116.png]]

## 5. 这几个函数怎么用

| 函数 | 作用 | 建议 |
| --- | --- | --- |
| `get_library_version()` | 返回 SDK 版本字符串 | 启动时打一条日志，排查“环境版本不对”最有用 |
| `set_log_level(level)` | 设置日志级别（Debug/Info/Warn/Error） | 开发期 Debug，发布前改 Warn/Error |
| `log_to_stderr()` | 日志输出到标准错误 | 命令行工具、容器里最方便 |
| `log_to_file(path)` | 日志写文件 | 设备上跑无人值守时用，方便事后查 |

## 6. 推荐初始化顺序

```cpp
#include "aimo/aimo.h"

int main() {
    aimo::log_to_stderr();                    // 先定输出方向
    aimo::set_log_level(aimo::LogLevel::kInfo);
    AIMOLOG_INFO("aimo version: %s", aimo::get_library_version());

    // … 之后再做 Model / InterpreterBuilder / invoke
}
```

## 7. 设备上排查问题的套路

1. 版本对不上：先看 `get_library_version()` 与模型转换工具版本是否匹配（模型格式通常不向后兼容）。
2. 推理结果全 0 / NaN：把 `set_log_level(Debug)` 打开，看张量形状与量化参数是否打印正确。
3. 偶发崩溃：用 `log_to_file` 留日志，定位是 `invoke` 还是内存越界。
4. 发布版：`log_to_file` 写到可写目录（不要写 `/` 或只读分区），并做日志轮转，否则会写满磁盘。

## 相关

- [[AIMO C++ API 的枚举类]] — LogLevel 取值
- [[AIMO API]]
- [[C++部署代码笔记]]
