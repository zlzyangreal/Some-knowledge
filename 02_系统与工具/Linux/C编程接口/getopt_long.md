# 1.原型
```c
int getopt_long(int argc, char *argv[],
                 const char *options, 
                 const struct option *long_options, 
                 int *optind);
```
参数说明：
- `argc` 和 `argv`：与 `main` 函数的参数相同，分别表示参数的个数和参数数组。
- `options`：一个字符串，定义了短选项字符集和它们是否需要参数。格式与 `getopt` 的 `optstring` 参数相同。(**我的理解就是设置结构体里面的val来找有不有**)
- `long_options`：一个  [[option结构体]] ，定义了长选项的名称、是否需要参数等信息。
- `optind`：一个指向整数的指针，用于存储下一个要解析的参数的位置。
返回值：
- 如果成功，`getopt_long` 返回匹配的短选项字符或长选项的 `val` 值。
- 如果没有更多选项，返回 `-1`。
- 如果遇到未知的选项或选项格式错误，返回 `':'`
# 2.例子
判断返回值
```c
    static struct option longOpts[] = {
        {"model_path", required_argument, nullptr, 'm'}, // -m --model_path
        {"label_path", required_argument, nullptr, 'l'},
        {"threads", required_argument, nullptr, 't'},
        {"camera_index", required_argument, nullptr, 'i'},
        {"width", required_argument, nullptr, 'w'},
        {"height", required_argument, nullptr, 'h'},
        {"fps", required_argument, nullptr, 'f'},
        {"help", no_argument, nullptr, '?'},
        {nullptr, 0, nullptr, 0}
    };

int c, optionIndex = 0;

c = getopt_long(argc, argv, "m:l:t:i:w:h:f:?", longOpts, &optionIndex)) != -1
```