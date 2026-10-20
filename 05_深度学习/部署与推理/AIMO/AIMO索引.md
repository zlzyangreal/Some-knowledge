# AIMO

广和通 AIMO 嵌入式推理框架与 C++ API。

上级：[[部署与推理索引]]　|　总索引：[[总索引]]

## 笔记（9）

- [[AIMO API]] — AIMO 的 C++ 接口是类 TFLite 风格的三层结构：
- [[AIMO C++ API 的 Config 类]] — 用于记录需要预先设置的配置选项，这些配置项在运行时会被用到。
- [[AIMO C++ API 的 Interpreter 类]] — 好像是解释器
- [[AIMO C++ API 的 InterpreterBuilder 类]] — 注意官方函数名里是 interpretper（拼写如此），写代码时照抄，别自己“纠正”成 interpreter。
- [[AIMO C++ API 的 Model 类]] — Model 类主要用于记录模型的文件信息、结构信息、运行过程中模型相关内容。
- [[AIMO C++ API 的其他函数]] — 版本对不上：先看 get_library_version() 与模型转换工具版本是否匹配（模型格式通常不向后兼容）。
- [[AIMO C++ API 的枚举类]] — int8 模型喂 float：不会报错，但结果完全乱——因为字节解释方式变了。
- [[AIMO使用]] — 网站，直接用手机号登陆
- [[C++部署代码笔记]] — std::vector<std::vector<uint32_t: 这是一个嵌套的vector，外层是一个vector，
