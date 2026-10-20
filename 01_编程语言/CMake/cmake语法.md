# 语法基本格式
语法采用: `指令(参数1 参数2 ...)`
* 参数之间用空格分开
* 指令不区分大小写
* 变量用`${}`取值,在IF语句中直接用变量名
# 重要指令
## [[基础指令]]
* `cmake_minimum_required`指定最小版本
* `project`定义工程名称,并可指定工程支持的语言
## [[链接与定义变量指令]]
* `set`显式的定义变量
* `add_executable`生成可执行文件
* `include_directories` 向工程添加多个特定的头文件搜索路径
* `install`定义安装规则
* `add_definitions`向编译器添加定义
* `file`将符合指定模式的文件列表存储到变量中
* `target_link_libraries` 为目标添加链接的共享库
## CMake 中用于输出信息
```cmake
message
# 输出一般消息。这将在构建过程中打印指定的文本
message("message text")

# 输出状态消息。这将在构建过程中打印指定的文本，并以状态消息的形式显示。
message(STATUS "message text")

# 输出警告消息。这将在构建过程中打印指定的文本，并以警告消息的形式显示。
message(WARNING "message text")

# 输出致命错误消息并终止构建。这将在构建过程中打印指定的文本，并以致命错误消息的形式显示。构建将在此处停止。
message(SEND_ERROR "message text")

# 输出详细信息。这将在构建过程中打印指定的文本，并以详细信息的形式显示。这些消息通常用于提供更详细的构建过程信息。
message(VERBOSE "message text")
```

## 条件判断
 **1.比较两个字符串是否相等**
* `EQUAL` 或 `STREQUAL`
```cmake
if (variable1 STREQUAL variable2)
endif()
```
**2.比较两个字符串是否不相等**
* `NOT EQUAL` 或 `STRNEQUAL`
**3.用于比较两个字符串的字典顺序，左边的字符串是否小于右边的字符串**
* `LESS` 或 `STRLESS`
**4.用于比较两个字符串的字典顺序，左边的字符串是否大于右边的字符串**
* `GREATER` 或 `STRGREATER`
**5.用于比较两个字符串的字典顺序，左边的字符串是否小于或等于右边的字符串**
* `LESS_EQUAL` 或 `STRLESS_EQUAL`
**6.用于比较两个字符串的字典顺序，左边的字符串是否大于或等于右边的字符串**
* `GREATER_EQUAL` 或 `STRGREATER_EQUA`
## 函数封装
function
```cmake
function(<name> [arg1 [arg2 ...]])
  # 一系列CMake命令
endfunction()
```
- `<name>` 是函数的名称，你可以通过这个名称来调用这个函数。
- `arg1`, `arg2`, ... 是函数的参数，可以在函数体内部使用这些参数。

使用`function`定义的代码块可以被其他CMake命令调用，例如：
```cmake
function(my_custom_function)   
	# 执行一些操作 
endfunction() 

# 调用函数 
my_custom_function()
```
## 向工程添加多个特定的库文件搜索路径
* 相当于g++ `-L`参数

`link_directories(dir1 dir2 ...)`
```cmake
link_directories(/usr/include/mylibfolder ./lib)
```
## 向当前工程添加存放源文件的子目录,并可以指定中间二进制和目标二进制存放的位置

`add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])`
```cmake
# 添加src子目录,src中需有一个CMakeLists.txt
add_subdirectory(src)
```
## 查找和引入外部依赖项
```cmake
find_package
# eg
find_package(Threads REQUIRED)
```
*  `Threads` 查找系统中可用的线程库
* `REQUIRED` 必须项
## 生成库文件

`add_library(libname [SHARED|STATIC|MODULE] [EXCLUDE_FROM_ALL] source1 source2 ...)`
```cmake
# 通过变量 SRC 生成 libhello.so 共享库
add_library(hello SHARED ${SRC})
```
* SHARED 动态库/ STATIC 静态库

## 添加编译参数

`add_complie_options(<option>...)`
```cmake
# 添加编译参数-wall -std=c++11
add_complie_options(-wall -std=c++11 -o2)
```
* -o2 是优化选项

## 发现目录下所有的源代码文件并将列表储存在一个变量中，这个指令临时被用来自动构建源文件列表

`aux_source_directory(dir VARIABLE)`
```cmake
# 定义 SRC 变量,其值为当前目录下所有的源代码文件
aux_source_dirctory(. SRC)
# 编译 SRC 变量所代表的源文件代码,生成 main 可执行文件
add_executable(main ${SRC})
```
## 设置目标的头文件包含路径
```cmake
target_include_directories
# eg
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```
* `PRIVATE` 私有
* `CMAKE_CURRENT_SOURCE_DIR` 见 [[cmake常用变量]]
