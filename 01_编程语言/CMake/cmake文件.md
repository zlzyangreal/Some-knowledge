# [[cmake工程构建]]
## [[CMakeLists.txt|**cmake文件模板**]]
## [[cmake调试设置指令语句]]
# [[cmake语法]]
# [[cmake常用变量]]
# 其他
## [[ccmake|可视化CMake文件]]
## [[OpenCV CMakeLists.txt链接]]

## 最小可用的 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.10)
project(demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 收集源文件
file(GLOB_RECURSE SRC ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(demo ${SRC})
target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 链接第三方库（以 OpenCV 为例）
find_package(OpenCV REQUIRED)
target_link_libraries(demo PRIVATE ${OpenCV_LIBS})
```

## 三条核心命令

1. `add_executable / add_library`：告诉 CMake 要产出什么，源码列表给它。
2. `target_include_directories`：头文件在哪（`PUBLIC` 传给依赖它的目标，`PRIVATE` 自己用）。
3. `target_link_libraries`：链接什么库。

`include_directories` / `link_directories` 这种“全局”写法是老风格，多目标时容易互相污染，
现在推荐 `target_*` 系。

## 两套构建目录

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # 配置
cmake --build build -j8                          # 编译
```

习惯上把 build 目录加进 `.gitignore`，源码树永远保持干净。

## 常用变量

- `CMAKE_SOURCE_DIR` / `CMAKE_BINARY_DIR`：工程根 / 构建根
- `PROJECT_NAME`：`project()` 里起的名字
- `CMAKE_CXX_FLAGS` / `CMAKE_BUILD_TYPE`：编译选项与构建类型（Debug 会带 `-g`）

## 相关

- [[cmake工程构建]] — 从零建工程
- [[基础指令]]、[[cmake语法]]、[[cmake常用变量]]
- [[链接与定义变量指令]]、[[cmake调试设置指令语句]]
- [[ccmake]] — 文本界面可视化改选项
- [[OpenCV CMakeLists.txt链接]]
