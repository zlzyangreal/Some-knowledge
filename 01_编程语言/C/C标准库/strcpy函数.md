`strcpy` 是一个在 C 语言中常用的字符串操作函数，用于将一个字符串复制到另一个字符串中。它定义在头文件 `<string.h>` 中
```c
char *strcpy(char *dest, const char *src);
```
`strcpy` 的作用是将源字符串 `src` 复制到目标字符串 `dest` 中。它会复制整个字符串，包括字符串的结尾字符 `\0`