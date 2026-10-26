# 1.结构体
## 1.[[option结构体]]
# 2.函数
## 1.[[getopt_long]]
# 3.全局变量
## 1.[[optarg]]

## 典型用法：getopt_long 解析命令行

```c
#include <stdio.h>
#include <getopt.h>     // getopt_long、struct option、optarg
#include <stdlib.h>

int main(int argc, char** argv) {
    int opt;
    int verbose = 0;
    const char* out = "a.out";

    static struct option long_opts[] = {
        {"verbose", no_argument,       0, 'v'},
        {"output",  required_argument, 0, 'o'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}                 // 必须以全 0 结尾
    };

    while ((opt = getopt_long(argc, argv, "vo:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'v': verbose = 1; break;
        case 'o': out = optarg; break;   // optarg 指向参数值
        case 'h':
        default:
            printf("用法: %s [-v] [-o file]\n", argv[0]);
            return 0;
        }
    }
    printf("verbose=%d output=%s\n", verbose, out);
    return 0;
}
```

## 三个要点

1. `struct option` 的四个字段：长选项名、是否有参数（`no_argument` / `required_argument` / `optional_argument`）、`flag`（一般填 0）、返回值（一般填短选项字符）。
2. `optarg` 是**全局变量**，指向当前选项的参数；只有 `required_argument` 才有值。
3. 选项解析完后 `optind` 指向第一个非选项参数，剩下的就是位置参数。

## 相关

- [[option结构体]]
- [[getopt_long]]、[[optarg]]
- [[C标准库]]
