好像是解释器
# 成员函数
## 1.1 init
![[Screenshot 2024-06-09 210517.png]]
## 1.2 load_model
![[Screenshot 2024-06-09 210749.png]]
## 1.3 set_input_tensor
![[Screenshot 2024-06-09 212200.png]]
## 1.4 invoke
![[Screenshot 2024-06-09 212647.png]]
## 1.5 get_output_tensor
![[Screenshot 2024-06-09 213100.png]]
## 1.6成员函数 create_instance
![[Screenshot 2024-06-16 172906.png]]
## 1.7成员函数 destory
![[Screenshot 2024-06-16 173219.png]]

## 2. 各成员函数说明

| 函数 | 作用 | 调用时机 |
| --- | --- | --- |
| `init()` | 初始化运行时环境（线程池、后端） | 构造后第一件事 |
| `load_model()` | 把模型图与权装载进内存 | `init()` 之后 |
| `set_input_tensor(idx, data, bytes)` | 把输入数据写进第 idx 个输入张量 | 每次推理前 |
| `invoke()` | 同步执行一次推理 | 输入就绪后 |
| `get_output_tensor(idx, buf, bytes)` | 读出第 idx 个输出张量 | `invoke()` 之后 |
| `create_instance()` | 创建同配置的另一个实例（多线程/多流） | 需要并发推理时 |
| `destory()` | 释放实例资源 | 退出前（注意官方拼写是 `destory`） |

## 3. 完整调用顺序

```cpp
aimo::InterpreterBuilder builder(&model, &config);
std::unique_ptr<aimo::Interpreter> interp;
builder(&interp);

interp->init();                 // ① 运行时初始化
interp->load_model();           // ② 装载模型

interp->set_input_tensor(0, input.data(), input.size() * sizeof(float));  // ③ 填输入
interp->invoke();               // ④ 推理

std::vector<float> out(OUT_SIZE);
interp->get_output_tensor(0, out.data(), out.size() * sizeof(float));     // ⑤ 取输出

interp->destory();              // ⑥ 释放
```

## 4. 三个容易踩的坑

1. **顺序不能错**：没 `init` 就 `load_model`、没 `set_input_tensor` 就 `invoke`，多数实现会返回错误码而不是崩溃，
   所以**一定要检查返回值**并在出错时打印。
2. **字节数**：`set_input_tensor` 的第三个参数是**字节数**。int8 模型用 `元素数 × 1`，float 用 `× 4`。
3. **`create_instance` 与多线程**：单个 Interpreter 不是线程安全的。要多路并发就每线程一个 instance，
   不要共用一个 `invoke`。

## 5. 典型封装

```cpp
class AimoRunner {
public:
    bool load(const std::string& path) {
        model_ = std::make_unique<aimo::Model>(path);
        config_ = std::make_unique<aimo::Config>();
        aimo::InterpreterBuilder b(model_.get(), config_.get());
        if (b(&interp_) != aimo::kOk) return false;
        if (interp_->init() != aimo::kOk) return false;
        return interp_->load_model() == aimo::kOk;
    }
    bool run(const std::vector<float>& in, std::vector<float>& out) {
        if (interp_->set_input_tensor(0, in.data(), in.size() * sizeof(float)) != aimo::kOk) return false;
        if (interp_->invoke() != aimo::kOk) return false;
        out.resize(OUT_SIZE);
        return interp_->get_output_tensor(0, out.data(), out.size() * sizeof(float)) == aimo::kOk;
    }
private:
    std::unique_ptr<aimo::Model> model_;
    std::unique_ptr<aimo::Config> config_;
    std::unique_ptr<aimo::Interpreter> interp_;
};
```

（返回码枚举名以官方头文件为准。）

## 相关

- [[AIMO C++ API 的 InterpreterBuilder 类]] — 怎么造出 Interpreter
- [[AIMO C++ API 的其他函数]] — 日志与版本
- [[C++部署代码笔记]] — 实际项目里的封装
