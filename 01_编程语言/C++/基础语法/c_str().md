`c_str` 是 `std::string` 类的一个成员函数，它返回一个指向以空字符终止的字符数组的指针，该数组包含与 `std::string` 对象相同的序列。这个函数特别用于与那些需要C风格字符串（即以null终止的字符数组）的函数或API进行交互。

例子
```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";
    const char* cstr = str.c_str();

    std::cout << cstr << std::endl; // 输出: Hello, World!

    // cstr 是一个指向C风格字符串的指针，不能修改它的内容
    // 下面的代码将导致编译错误，因为cstr是const的
    // cstr[0] = 'h';

    return 0;
}
```