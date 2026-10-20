`optarg` 是一个全局变量，它在 POSIX 兼容的系统（包括大多数 Linux 和 UNIX 系统）中定义，用于存储由 `getopt` 和 [[getopt_long]] 函数解析的命令行选项的参数值

**例子**
```cpp
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
    while ((c = getopt_long(argc, argv, "m:l:t:i:w:h:f:?", longOpts, &optionIndex)) != -1) {
        switch (c) {
        case 'm':
            options.model_path = optarg;
            break;
         ...
         }
    }
```