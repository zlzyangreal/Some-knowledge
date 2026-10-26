# 标准库函数
## 1.1字符串处理
### 1.1.1[[strtod函数]]
### 1.1.2[[atoi函数]]
## 1.2数学库algorithm
### 1.2.1[[sqrt函数]]
### 1.2.2[[pow函数]]
### 1.2.3 [[min_element函数与max_element函数]]
## 2.STL
### 2.1[[push_back函数]]
## 3.C++11
### 3.1[[emplace_back函数]]
## 4.C++ 14
### 4.1[[make_unique函数模板]]
# STL库
## 1.**容器适配器**
### 1.1[[queue]]

## 5. 导读：怎么用这张目录

上面这棵树是“按我踩过的坑”组织的，不是按标准文档组织的。真正查函数签名建议直接开
[cppreference](https://zh.cppreference.com/w/cpp) ，这里只记“什么时候该想起它”。

### 常打交道的头文件

| 头文件 | 里面有什么 | 典型用途 |
| --- | --- | --- |
| `<string>` | `std::string`、`to_string`、`stoi` | 字符串拼接、数字互转 |
| `<vector>` / `<array>` | 动态/静态数组 | 绝大多数场景的首选容器 |
| `<map>` / `<unordered_map>` | 红黑树 / 哈希表 | 计数、查表 |
| `<set>` / `<unordered_set>` | 去重集合 | 判重、去重 |
| `<algorithm>` | `sort`、`find`、`min_element`、`count_if` | 排序查找 |
| `<numeric>` | `accumulate`、`iota` | 求和、初始化序列 |
| `<sstream>` | `stringstream` | 字符串切分、格式化 |
| `<memory>` | `unique_ptr`、`shared_ptr`、`make_unique` | 管堆内存，替代裸 new/delete |
| `<thread>` / `<mutex>` / `<atomic>` | 并发 | 见 [[C++多线程]] |

### 记忆顺序

1. 先想**容器**（vector/map/set/queue）——90% 的数据结构需求都在这里。
2. 再想**算法**（sort/find/count_if）——不要手写排序。
3. 最后才想**语言特性**（lambda、模板、智能指针）。

### 关于 `queue` 这类容器适配器

`stack` / `queue` / `priority_queue` 不是容器，而是**在 `deque`/`vector` 上包装出的接口**（适配器），
所以它们没有迭代器、不能遍历——只能 `push/pop/top`。需要遍历就换 `deque` 或 `list`。

## 相关

- [[容器]]、[[queue]]、[[set容器]]、[[vector类]]
- [[algorithm库]]、[[stringstream的用法]]
- [[make_unique函数模板]]、[[new and delete]]
