# 函数原型

```cpp
std::string substr(size_t pos = 0, size_t count = npos) const;
```
- **`pos`**：子串的起始位置（从 0 开始）。
- **`count`**：要截取的长度，默认是 `npos`（即直到字符串末尾）。
- **返回值**：返回一个新的 `std::string` 对象，不会修改原字符串。
如果 `pos > size()`，会抛出 `std::out_of_range` 异常。