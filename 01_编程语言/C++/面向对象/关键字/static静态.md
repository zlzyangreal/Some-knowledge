# 1、C++的 static 保留了C的特性
- [[C语言中的static]]
# 2、**修饰类成员变量**
静态成员变量属于类，而不是属于某个特定对象。所有对象共享同一个静态成员变量。
-  **生命周期**：静态成员变量在类加载时被初始化一次，其生命周期是整个程序的运行期间。
- **初始化方式**：静态成员变量可以在类定义外部初始化，也可以在类内使用 constexpr 初始化（C++11 及以后版本）。
```cpp
class MyClass {
public:
    static int count;
};

int MyClass::count = 0;  // 类外初始化
```
```cpp
// or
class MyClass {
public:
    static constexpr int count = 0;  // 类内初始化（C++11 及以后版本）
};
```
# 3.**作用域与静态函数调用**
```cpp
class SubarrayChecker {
};
//////////////////////////
auto parsed = SubarrayChecker::parseInput(input);
if (SubarrayChecker::check(parsed)) {
    std::cout << "true\n";
}
```
**主函数没有声明对象也能直接用类名调用函数**，是因为：
`parseInput` 和 `check` 在类里被声明为 `static` 成员函数。
- `static` 成员函数属于 **类本身**，不依赖于某个对象实例。
- 因此可以用 `类名::函数名(...)` 的方式直接调用。