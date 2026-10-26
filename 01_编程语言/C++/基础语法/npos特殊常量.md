在 C++ 标准库里，`std::string::npos` 表示 **无效位置**
```cpp
std::string s = "hello";
auto pos = s.find('x');  // 找不到 'x'
if (pos == std::string::npos) {
    std::cout << "没找到\n";
}
```