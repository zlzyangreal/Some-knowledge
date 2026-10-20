`noexcept` 的意思是：**这个函数不会抛出异常**
```cpp
void f() noexcept {
    // 这里面不能抛异常
}
```
如果 `f` 在运行时真的抛出了异常，程序会直接调用 `std::terminate()` 结束，而不会进入 `catch`
# 1.用法
## 1.1固定 `noexcept`
```cpp
void func() noexcept;   // 承诺绝不会抛异常
```
## 1.2条件 `noexcept`
```cpp
void func() noexcept(expression);
```
其中 `expression` 是一个常量布尔表达式：
- 如果为 `true`，函数被标记为不会抛异常；
- 如果为 `false`，函数可能抛异常。

