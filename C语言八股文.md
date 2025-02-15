# 一、宏定义
## 1. 用预处理指令#define 声明一个常数，用以表明1年中有多少秒（忽略闰年问题）
```c
#define  SECONDS_PER_YEAR (365*24*60*60)UL
```
- [UL的意义](C语言define使用)
## 2. 写一个"标准"宏MIN ，这个宏输入两个参数并返回较小的一个
```c
#define MIN(A,B) ((A)<=(B)?(A):(B))

int main(){
	int a = 0;
	int b = 1;
	int min_data = MIN(a,b);
}
```
## 3. 预处理器标识#error的目的是什么？
编译时检查某些条件，如果这些条件不满足，则立即停止编译并输出错误信息。
```c
// 检查 MY_MACRO 是否被定义
#ifndef MY_MACRO
#error "MY_MACRO must be defined"
#endif

// 如果 MY_MACRO 被定义，继续编译
int main() {
    // 程序的其他部分
    return 0;
}
```
## 4. 嵌入式系统中经常要用到无限循环，你怎么样用C编写死循环呢？
```c
int main(){
	while(1){
		
	}
}
```
## 5.已知数组table，用宏求数组元素个数
```c
#define COUNT(table) (sizeof(table) / sizeof(table[0]))
```
## 6.带参宏和函数的区别？
- 带参宏只是在编译预处理阶段进行简单的字符替换；而函数则是在运行时进行调用和返回。
- 宏替换不占运行时间，只占编译时间；而函数调用则占运行时间
- 带参宏在处理时不分配内存；而函数调用会分配临时内存
- 宏不存在类型问题，宏名无类型，它的参数也是无类型的；而函数中的实参(实际参数)和形参(形式参数)都要定义类型，二者的类型要求一致
```C
int add(a,b){ //a b 是形参
	int c =0;
	c = a + b;
	return c;
}

int main(){
	int num = add(2,3); //2 3 是实参
}
```
## 7.内联函数的优缺点和适用场景是什么？
- 优点：内联函数与宏定义一样会在原地展开，省去了函数调用开销，同时又能做类型检查
- 缺点：它会使程序的代码量增大，消耗更多内存空间
## 8.如何用C语言实现读写寄存器变量？
```c
#define rBANKCON0 (*(volatile unsigned long *)0x48000004)
rBankCON0 = 0x12;
```
- 由于是寄存器地址，所以需要先将其强制类型转换为 volatile unsigned long *
## 9.下面代码能不能编译通过？
```c
#define c 3
c++;
```
- 不能。c宏定义是常数，++用于变量
## 10.在C语言中，凡是以#开头的都是预处理命令，同时预处理命令都是以#开头的

# 二、关键字
## 1.关键字 static 的作用是什么？
- 1、**修饰局部变量**：静态局部变量的作用域仍然是所在的函数内部，其他函数无法访问。
- 2、**修饰全局变量**：静态全局变量的作用域仅限于定义它的源文件内。其他源文件无法直接访问该静态全局变量，即使通过外部变量声明（`extern`）也无法访问。
- 3、**修饰函数**：被 `static` 修饰的函数称为静态函数，其作用域仅限于定义它的源文件内。其他源文件无法直接调用该静态函数，即使通过函数指针等方式也无法调用。
- [实例](C语言中的static)
## 2.关键字volatile的作用是什么？
- 作用：**告诉编译器不要去假设（优化）这个变量的值**，因为这个变量可能会被意想不到地改变。精确地说就是，**优化器在用到这个变量时必须每次都小心地重新读取这个变量的值，而不是使用保存在寄存器里的备份**
①并行设备的硬件寄存器（如：状态寄存器）。
②一个中断服务子程序中会访问到的非自动变量。
③多线程应用中被几个线程共享的变量（防止死锁）。
## 3.关键字const的使用
`const` 关键字在 C 和 C++ 中用于声明常量或指定变量的值在初始化后不能被修改。
```c
const int a; // a是一个整形常量
int const a; // a是一个整形常量

const int *a; // a是一个指向整型常量的指针变量
int * const a; // a是一个指向整型变量的指针常量
int const * const a = &b; // a是一个指向整型常量的指针常量

char *strcpy(char *strDest, const char *strSrc); // 参数在函数内部不会被修改
const int strcmp(char *source, char *dest); // 函数的返回值不能被修改
```
- [[char类型]]
- [[strcpy函数]]
## 4.关键字typedef
`typedef` 是 C 和 C++ 中的一个关键字，用于为已有的类型创建一个新的名字（别名）。它可以帮助简化复杂的类型声明
```c
typedef int Integer;
typedef float Real;

Integer a = 10;  // 等价于 int a = 10;
Real b = 3.14;   // 等价于 float b = 3.14;
```
## 5.关键字sizeof的作用是什么？函数strlen()呢？
- sizeof关键字用来计算变量、数据类型所占内存的字节数。sizeof(数组名)得到数组所占字节数，sizeof(字符串指针名)得到指针所占字节数。
- 而strlen()函数则用来测试字符串所占字节数，不包括结束字符 ’\0’。strlen(字符数组名)得到字符串所占字节数，strlen(字符串指针名)得到字符串所占字节数。
## 6.关键字extern的作用是什么？
用于跨文件引用全局变量，即在本文件中引用一个已经在其他文件中定义的全局变量
- 注意引用时不能初始化，如extern var，而不能是extern var = 0
- 另外，函数默认是extern类型的，表明是整个程序（工程）可见的，加不加都一样
## 7.extern”C”的作用？
在C++代码中调用C函数，用法：extern “C”{C函数库头文件/函数声明}
## 8.关键字auto的作用是什么？
这个变量在C和C++11后有区别
- 1. 在 C 语言中，`auto` 是一个存储类别说明符，用于显式声明变量为“自动存储期”（auto storage duration）的变量。在 C 语言中，局部变量默认就是自动存储期的
- **2.`auto` 让编译器根据变量的初始化表达式自动推导变量的类型。**
```c++
auto a = 10;  // a 的类型被推导为 int
auto b = 3.14;  // b 的类型被推导为 double
auto c = "Hello";  // c 的类型被推导为 const char*

auto& ref = a;  // ref 是 int& 类型
auto* ptr = &a;  // ptr 是 int* 类型

auto arr = new int[10];  // arr 是 int* 类型
```
## 9.关键字register的作用是什么？使用时需要注意什么？
编译器会将register修饰的变量尽可能地放在CPU的寄存器中，以加快其存取速度，一般用于频繁使用的变量。
- 只是建议，不一定会照做
- 在C++17已经弃用，C++20已经移除
## 10.C语言编译过程中，关键字volatile和extern分别在哪个阶段起作用？
volatile在编译阶段，extern在链接阶段
- C语言编译过程分为预处理、编译、汇编、链接
## 11.const与#define的异同？
前者有数据类型，两者都可以定义常数
# 三、数据结构
## 1. 用变量a给出下面的定义

a) 一个整型数（An integer）  
b)一个指向整型数的指针（ A pointer to an integer）  
c)一个指向指针的的指针，它指向的指针是指向一个整型数（ A pointer to a pointer to an intege）r  
d)一个有10个整型数的数组（ An array of 10 integers）
e) 一个有10个指针的数组，该指针是指向一个整型数的。（An array of 10 pointers to integers）f) 一个指向有10个整型数数组的指针（ A pointer to an array of 10 integers）
g) 一个指向函数的指针，该函数有一个整型参数并返回一个整型数（A pointer to a function that takes an integer as an argument and returns an integer）
h) 一个有10个指针的数组，该指针指向一个函数，该函数有一个整型参数并返回一个整型数（ An array of ten pointers to functions that take an integer argument and return an integer ）
```c
// a
int A_data = 0; 
// b
int *a; //指针 `a` 并没有被初始化，因此它是一个野指针（wild pointer），指向一个不确定的内存地址。
// c
int **a; //二级指针
// d
int A_arr[10];
// e
int *a[10];
// f
int (*a)[10];
// g
int (*a)(int);
// h
int (*a[10])(int);
```
- [[指针型定义]]
- [[二级指针]]
## 2.下面的代码输出是什么，为什么？
```c
void foo(void) {
	unsigned int a = 6;
	int b = -20;
	(a + b > 6)? printf("> 6") : printf(" <= 6");
}
```
- 输出>6
## 3.写出float x与“零值”比较的if语句。
```c
if(x>-0.01 && x<0.01);
```
- 浮点数不能用`==` `!=`
## 4.下面代码有什么错误？
```c
#include<stdio.h>
void main() {
	char *s = "AAA";
	s[0] = 'B';
	printf("%s", s);
}
```
- s是指针，直接指向了常量
- s在常量的基础上直接赋值是不合法的
## 5.下面代码输出是什么？
```c
#include<stdio.h>
void main() {
	int *a = (int *)2;
	printf("%d", a + 3);
}
```
- 第一步将2指针化，在指针的基础上加3，由于`int`是4字节，相当于`2+3*4`
## 6.下面代码运行后会是什么现象？
```c
#include<stdio.h>

#define N 500
void main() {
	unsigned char count;
	for(count = 0; count < N; count++) {
		printf("---%d---\n", count);
	}
}
```
- `unsigned char`最大为255，会一直打印从0-255
## 7.下面函数的返回值是？
```c
int foo(void) {
	int i;
	char c = 0x80;
	i = c;
	if(i > 0)
		return 1;
	return 2;
}
```
- char范围是(`-128`-`127`),`0x80`是128超了，`i=-128`返回2
## 8.结构体内存对齐原则？
1. 第一个成员的首地址（地址偏移量）为0
2. **对齐单位**：每个成员变量的对齐单位通常是其类型大小的整数倍
- `char` 类型的对齐单位是1字节
- `int` 类型（通常是4字节）的对齐单位是4字节
- `double` 类型（通常是8字节）的对齐单位是8字节
3. **结构体的总大小必须是最大对齐单位的整数倍**
```c
#include <stdio.h>

struct Example {
    char a;        // 1字节，对齐单位为1
    int b;         // 4字节，对齐单位为4
    double c;      // 8字节，对齐单位为8
};

int main() {
    printf("Size of struct Example: %zu\n", sizeof(struct Example));
    return 0;
}
```
- 输出`Size of struct Example: 16`
## 9.结构体内存对齐的原因？
1. 平台原因（移植原因）：不是所有的硬件平台都能访问任意地址上的任意数据
2. 对齐的内存访问通常比非对齐访问更快
## 10.给定结构体，它在内存中占用多少字节（32位编译器）？
```c
struct A {
	char t : 4;// 4位
	char k : 4;// 4位
	unsigned short i : 8; // 8位
	unsigned long m; // 4字节
};
```
- **1字节（Byte）等于8位（Bit）**
- 结构体8字节
## 11.在32位系统中，有如下结构体，那么sizeof(fun)的数值是？
```c
#pragma pack(1)
struct fun {
	int i; // 4字节
	double d; // 8字节
	char c; // 1字节
};
```
- `sizeof`返回字节数，`pragma pack(1)`修改对其字节数为1，返回13
## 12.数组首元素地址和数组地址的异同？
1. 异：数组首元素地址和数组地址是两个不同的概念。例如`int a[10]`，a的值是数组首元素地址，所以a+1就是第二个元素的地址，int类型占用4个字节，所以两者相差4。而&a是数组地址，所以&a+1就是向后移动`(10*4)`个单位，所以两者相差40。
2. 同：数组首元素地址和数组地址的值是相等的。
## 13.下面代码输出是什么？
```c
#include<stdio.h>

void main() {
	int a[5] = {1, 2, 3, 4, 5};
	int *ptr = (int *)(&a + 1);
	printf("%d, %d", *(a + 1), *(ptr - 1));
}
```
- a是首元素地址，`*(a + 1)`为`a[1]`
- `&a`是数组地址，`&a +1`是数组结尾后下一个数组长的数组的地址，`*(ptr -1)`是`a[4]`
## 14.判断下列表达式正确与否？
```c
char str1[2][3] = {“a”, “b”};
char str2[2][3] = {{1, 2}, {3, 4}, {5, 6}}; 
char str3[] = {“a”, “b”}; 
char str4[2] = {“a”, “b”}; 
```
-  正确，str1是一个可存放两个字符串的字符串数组
-  错误，行列不匹配,两行三列
-  错误，字符数组不能存放字符串
-  错误，字符数组不能存放字符串
- `'a'`是字符，`"a"`是字符串
## 15.查看下面代码，`p[6]`等于几？
```c
int a[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
int *p = &a[1];
```
- p指针指向`a[1]`，`p[6]`的意思是向后移6个`int`
- 输出为8
## 16.下面代码的输出结果是什么？
```c
#include<stdio.h>

void main() {
	char *str[] = {"ab", "cd", "ef", "gh", "ij", "kl"}; //指针数组
	char *t;
	t = (str + 4)[-1];
	printf("%s", t);
}
```
- `str`是一个指针数组，指向不同的字符串
- `str + 4`表示str数组的首地址向后移四位，即第五个元素
- `(str + 4)[-1]`表示在上述基础上向左移一位，第四个元素
## 17.变长数组是什么？
变长数组（Variable Length Array，简称 VLA）是C语言中一种特殊的数组类型，**它的大小可以在运行时动态确定，而不是在编译时固定**。变长数组是C99标准引入的一个特性，但在C11标准中被标记为可选特性，因此并不是所有编译器都支持它。
## 18.bool类型包含于哪个头文件？
`stdbool.h` 是C99标准引入的一个头文件，专门用于定义布尔类型
## 19.结构体struct和联合体union的区别？
- 结构体是一种用户自定义的数据类型，用于将多个不同类型的数据项组合在一起。每个数据项称为**成员（member）**，**每个成员在内存中占据独立的空间**。
```c
#include <stdio.h>

// 声明一个结构体
struct Student {
    int id;
    char name[50];
    float gpa;
};

int main() {
    // 创建一个结构体变量
    struct Student s1;

    // 初始化结构体成员
    s1.id = 101;
    strcpy(s1.name, "Alice");
    s1.gpa = 3.5;

    // 访问结构体成员
    printf("ID: %d, Name: %s, GPA: %.2f\n", s1.id, s1.name, s1.gpa);

    return 0;
}
```
假设结构体 `Student` 的内存布局如下：
1.  `int id`：占用 4 字节
2. `char name[50]`：占用 50 字节
3. `float gpa`：占用 4 字节
4.  **总大小**：至少 58 字节（可能因内存对齐而更大）
- 联合体也是一种用户自定义的数据类型，用于将多个不同类型的数据项组合在一起。但与结构体不同，**联合体的所有成员共享同一块内存空间**。
```c
#include <stdio.h>

// 声明一个联合体
union Data {
    int i;
    float f;
    char str[20];
};

int main() {
    // 创建一个联合体变量
    union Data data;

    // 初始化联合体成员
    data.i = 10;  // 存储整数
    printf("data.i: %d\n", data.i);

    data.f = 2.5;  // 存储浮点数
    printf("data.f: %.2f\n", data.f);

    strcpy(data.str, "Hello");  // 存储字符串
    printf("data.str: %s\n", data.str);

    return 0;
}
```
假设联合体 `Data` 的内存布局如下：
1. `int i`：占用 4 字节
2. `float f`：占用 4 字节
3. `char str[20]`：占用 20 字节
4. **联合体大小**：20 字节（最大成员的大小）
由于所有成员共享同一块内存，因此：
1. 当 `data.i` 被赋值时，`data.f` 和 `data.str` 的内容也会被覆盖。
2. 当 `data.str` 被赋值时，`data.i` 和 `data.f` 的内容也会被覆盖。

**两者最大的区别在于内存的使用。**
## 20.给了一个地址a，分别强转类型为：int变量、int指针、数组指针、指针数组、函数指针
```c
(int)a;
(int *)a;
(int (*)[]a);
(int *[]a);
(int (*)(int))a;
```
## 21.执行完下面代码，c的值是多少？
```c
unsigned int a = 1;
int b = 0;
int c = 0;
c = a + b > 0 ? 1 : 2;
```
- 输出1，无符号和有符号运算，有符号转换成无符号
## 22.C语言中不同数据类型之间的赋值规则？
1. 整数与整数之间（char, short, int, long）：
- **长度相等**：内存中的数据不变，只是按不同的编码格式来解析。
- **长赋值给短**：**截取低位**，然后按短整数的数据类型解析。
- **短赋值给长**：如果都是**无符号数**，**短整数高位补0**；如果都是**有符号数**，短整数**高位补符号数**；如果**一个有符号一个无符号**，那么先将短整数进行位数扩展，过程中保持数据不变，然后按照长整数的数据类型解析数据。
2. 整数与浮点数之间
- **浮点数转整数**：截取整数部分
- **整数转浮点数**：小数部分为0，整数部分与整数相等
- **会有精度丢失，所以不能来回转**
3. float与double之间
- double转float**会**丢失精度
- float转double**不会**丢失精度
# 四、内存管理
## 1.由gcc编译的C语言程序占用的内存分为哪几个部分？

| 栈区(stack)       | 存放函数的参数、局部变量。                                                                             |
| --------------- | ----------------------------------------------------------------------------------------- |
| 堆区(heap)        | 提供程序员动态申请的内存空间。                                                                           |
| 全局（静态）区(static) | 存放全局变量和静态变量，初始化不为0的全局变量和静态变量、const型常量在一块区域（.data段），未初始化的、初始化为0的全局变量和静态变量在相邻的另一块区域（.bss段）。 |
| 程序代码区           | 存放函数体的二进制代码和字符串常量                                                                         |
## 2.小端和大端
- 小端：一个数据的低位字节数据存储在低地址
- 大端：一个数据的高位字节数据存储在低地址
## 3.如何判读一个系统的大小端存储模式？
**方法1：使用指针**
通过将一个多字节的整数赋值给一个变量，并通过指针访问其最低地址的内容来判断。
```c
#include <stdio.h>

int main() {
    unsigned int x = 1; // 1 的二进制表示为 00000001 00000000 00000000 00000000
    char *c = (char*)&x; // 将 x 的地址转换为字符指针

    if (*c) { // 如果最低地址的内容为 1，则为小端
        printf("Little-Endian\n");
    } else { // 如果最低地址的内容为 0，则为大端
        printf("Big-Endian\n");
    }

    return 0;
}
```
**方法2：使用位运算**
通过位运算和联合体（`union`）来判断大小端模式。
```c
#include <stdio.h>

int main() {
    union {
        unsigned int i;
        char c[sizeof(unsigned int)];
    } u;

    u.i = 1; // 将 1 赋值给整数部分

    if (u.c[0]) { // 如果最低地址的内容为 1，则为小端
        printf("Little-Endian\n");
    } else { // 如果最低地址的内容为 0，则为大端
        printf("Big-Endian\n");
    }

    return 0;
}
```
## 4.全局变量和局部变量的区别？
- 全局变量储存在静态区，进入main函数之前就被创建，生命周期为整个源程序。
- 局部变量在栈中分配，在函数被调用时才被创建，在函数退出时销毁，生命周期为函数内。
## 5.以下程序中，主函数能否成功申请到内存空间？
```c
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

void getmemory(char *p) {
	p = (char *)malloc(100);
	strcpy(p, "hello world");
}

int main() {
	char *str = NULL;
	getmemory(str);
	printf("%s\n", str);
	free(str);
	return 0;
}
```
- 不能 ：`getmemory(str)`没能改变str的值，因为传递给子函数的只是str的复制值NULL，main函数中的str一直都是 NULL
```c
//修正
//传递的是二重指针，即str的指针
void getmemory(char **p) {
	*p = (char *)malloc(100);
	strcpy(*p, "hello world");
}
//传递的是指针别名，即str的别名，C++中
void getmemory(char * &p) {
	p = (char *)malloc(100);
	strcpy(p, "hello world");
}
```
## 6.运行下面的Test()函数会有什么样的后果？
```c
void GetMemory(char **p, int num) {
	*p = (char *)malloc(num);
}

void Test(void) {
	char *str = NULL;
	GetMemory(&str, 100);
	strcpy(str, "hello");
	printf("%s\n", str);
}
```
- 内存泄漏，没释放`free()`
## 7.运行下面的Test()函数会有什么样的后果？
```c
char *GetMemory(void) {
	char p[] = "hello world";
	return p;
}

void Test(void) {
	char *str = NULL;
	str = GetMemory();
	printf("%s\n", str);
}
```
- 在`GetMemory`函数里面定义了一个局部变量，在运行结束后被释放，返回了一个野指针
## 8.运行下面的Test()函数会有什么样的后果？
```c
void Test(void) {
	char *str = (char *) malloc(100);
	strcpy(str,"hello");
	free(str);
	if(str != NULL) {
	strcpy(str, "world");
	printf("%s\n", str);
	}
}
```
- 不能对已经释放了的指针继续操作
- 而且没有`str = NULL`会变成野指针
## 9.在C语言中memcpy和memmove是一样的吗？
- memcpy()与memmove()一样都是用来拷贝src所指向内存内容前n个字节到dest所指的地址上
- 不同的是，当src和dest所指的内存区域重叠时，memcpy可能无法正确处理，而memmove()仍然可以正确处理，不过执行效率上略慢些
## 10.malloc的底层实现？
1. **通过 `brk()` 系统调用**：
    - `brk()` 用于调整进程的堆顶指针，从而扩展堆空间。
    - 当请求的内存小于一定阈值（通常是 128KB）时，`malloc` 会通过 `brk()` 从堆中分配内存。
    - 初始时，堆空间大小为零，`brk` 指针指向堆的起始位置。随着内存分配请求，`brk` 指针向高地址移动。
2. **通过 `mmap()` 系统调用**：
    - `mmap()` 用于将文件或匿名内存映射到进程的地址空间。
    - 当请求的内存大于阈值（通常是 128KB）时，`malloc` 会通过 `mmap()` 分配内存。
    - `mmap()` 分配的内存不会立即占用物理内存，直到首次访问时才会触发缺页异常，由操作系统分配物理页面
## 11.在1G内存的计算机中能否malloc(1.2G)？为什么？
因为malloc函数是在程序的虚拟地址空间申请的内存，与物理内存没有直接的关系。虚拟地址与物理地址之间的映射是由操作系统完成的，操作系统可通过虚拟内存技术扩大内存。
