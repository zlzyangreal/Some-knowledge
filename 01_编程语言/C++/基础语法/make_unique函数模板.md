`std::make_unique` 是标准库中新增的一个函数模板，用于简化动态分配对象的构造过程，并避免使用 `new` 操作符时的内存泄漏。`std::make_unique` 函数自动处理了内存分配和对象构造，并返回一个指向新创建对象的 `std::unique_ptr`。
# 1.原型
```cpp
template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args);
```
参数说明：
- `T`：要创建的对象的类型。
- `Args`：传递给 `T` 的构造函数的参数。
返回值：
- 返回一个 `std::unique_ptr<T>`，指向用给定参数构造的对象。
# 2.优势
1. **类型安全**：`std::unique_ptr` 确保了类型安全，并且在对象生命周期结束时自动释放内存。
2. **异常安全**：如果对象构造过程中抛出异常，`std::make_unique` 会处理异常，不会泄露内存。
3. **简洁性**：避免了冗长的 `new` 表达式和 `std::unique_ptr` 的手动创建。
# 3.例子
```cpp
#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass(int x) : value(x) {
        std::cout << "MyClass(" << value << ") created" << std::endl;
    }
    ~MyClass() {
        std::cout << "MyClass(" << value << ") destroyed" << std::endl;
    }
    int value;
};

int main() {
    // 使用 make_unique 创建 MyClass 的实例
    auto ptr = std::make_unique<MyClass>(10);

    // 使用 unique_ptr 管理的 MyClass 实例
    ptr->value = 20;

    // 不需要手动释放内存，unique_ptr 析构时会自动处理
    return 0;
}
```
`std::make_unique` 用于创建 `MyClass` 的一个实例，并自动包装在一个 `std::unique_ptr` 中。这样，我们就不需要手动使用 `new` 来分配内存，也不需要担心在 `main` 函数结束时释放内存，因为 `std::unique_ptr` 在超出作用域时会自动释放它所管理的对象。
# 4.注意
`std::make_unique` 没有对应的数组版本，如果需要创建对象数组，仍然需要使用 `new` 和 `std::unique_ptr` 的数组版本。此外，`std::make_unique` 保证不会抛出异常（no-throw guarantee），这是通过在内存分配失败时返回 `nullptr` 来实现的。