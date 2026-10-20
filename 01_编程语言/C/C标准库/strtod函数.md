用于将字符串转换为双精度浮点数（`double` 类型）
# 1.原型
```c
double strtod(const char *str, char **endptr);
```
参数说明：
- `str`：指向要转换的字符串的指针。该字符串应该包含一个可能的符号（'+' 或 '-'），后跟一组数字（可以是整数或浮点数），并且可能包含一个小数点。
- `endptr`：一个指向字符指针的指针。如果提供了这个参数（即它不是 `NULL`），`strtod` 函数会在转换结束后，将这个指针设置为字符串中第一个未被转换的字符的位置。这允许调用者了解转换过程的结束位置。
返回值：
- `strtod` 返回转换后的双精度浮点数值。
- 如果字符串中没有有效的数字，它将返回 0.0。
- 如果转换的数太大或太小，无法表示为 `double` 类型，它将返回正负无穷大（`+/- HUGE_VAL`），并且 `errno` 将被设置为 `ERANGE`。
# 2.例子
```c
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int main() {
    const char *str = "123.45abc";
    char *endptr;
    double value;

    value = strtod(str, &endptr);

    if (endptr == str) {
        printf("No valid number found.\n");
    } else {
        printf("Converted value: %f\n", value);
        printf("Remaining string: \"%s\"\n", endptr);
    }

    return 0;
}
```
转换为 `double` 类型的数值 `123.45`。`endptr` 将指向字符串中的第一个未被转换的字符，即 `"abc"`