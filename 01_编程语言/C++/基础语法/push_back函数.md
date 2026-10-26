`push_back` 是 C++ 标准模板库（STL）中容器类的一个成员函数，用于在容器的末尾添加一个元素。这个函数是 `std::vector`、`std::list`、`std::deque` 等序列容器的通用操作。
# 1.原型
```cpp
void push_back(const value_type& value);
```
**对于移动语义支持的 C++11 及以上版本**
```cpp
void push_back(value_type&& value); // rvalue 引用版本
```
参数说明：
- `value`: 要添加到容器末尾的元素。
# 2.例子
```cpp
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v;

    // 添加元素到 vector
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);

    // 打印 vector 的内容
    for (int i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    return 0;
}
```
这段代码将输出：`10 20 30`
# 3.注意事项
1. **性能**：`push_back` 通常是一个高效的操作，但如果容器的当前容量不足以容纳新元素，它可能会触发容量增加和元素重新分配，这将是一个较昂贵的操作。
2. **元素复制**：当使用 `push_back` 添加一个元素时，如果元素类型是复制构造的，那么会复制传入的值。
3. **移动语义**：C++11 引入了右值引用和移动构造函数，允许使用 `push_back` 移动元素而不是复制，这可以提高性能，特别是在添加大型对象时。
4. **容器类型**：`push_back` 也适用于其他类型的容器，如 `std::list` 和 `std::deque`，但它们的行为可能略有不同，因为它们不是连续存储的。