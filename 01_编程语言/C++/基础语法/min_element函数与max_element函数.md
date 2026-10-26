用于**查找指定范围内的最小元素**
在给定的迭代器范围 `[first, last)` 中，找到指向最小元素的迭代器（左闭右开区间，即包含 `first` 指向的元素，不包含 `last` 指向的元素）。
# 例子
寻找数组最小值
```cpp
 auto min_it = std::min_element(num.begin(), num.end());
```
# max_element函数
反之，一样的用法