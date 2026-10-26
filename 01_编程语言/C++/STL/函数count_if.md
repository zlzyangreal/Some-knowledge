`std::count_if` 是标准库算法，用于统计满足条件的元素数量
# 1.作用
用于**统计容器中满足特定条件的元素个数**。
# 2.原函数
```cpp
template< class InputIt, class UnaryPredicate > typename iterator_traits<InputIt>::difference_type count_if( InputIt first, InputIt last, UnaryPredicate p );
```
输入：
1. `first` 和 `last`：输入迭代器，指定要遍历的元素范围（`[first, last)`，包含 `first` 指向的元素，不包含 `last` 指向的元素）。
2. - `p`：一元谓词（UnaryPredicate），即一个接受单个元素作为参数并返回 `bool` 类型的函数（或函数对象、lambda 表达式），用于判断元素是否满足条件。
返回值：
- 满足条件的元素个数（类型为迭代器差值类型，通常可视为 `int`）。
# 3.eg统计其中**偶数的个数**
```cpp
#include <iostream>
#include <vector>
#include <algorithm> // 包含 std::count_if

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6, 7, 8};
    
    // 统计偶数的个数：使用 lambda 表达式作为判断条件
    int evenCount = std::count_if(
        nums.begin(),    // 起始迭代器
        nums.end(),      // 结束迭代器
        [](int num) { return num % 2 == 0; } // 条件：是偶数
    );
    
    std::cout << "偶数的个数：" << evenCount << std::endl; // 输出：4
    return 0;
}
```