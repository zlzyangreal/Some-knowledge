## for (char ch : input)
```cpp
 int main() {
     std::string input = "Hello";
     for (char ch : input) {
         std::cout << ch << std::endl;
     }
     return 0;
}
```
表示从头开始遍历，按容器自然顺序
### 输出
```bash
H
e
l
l
o
```
