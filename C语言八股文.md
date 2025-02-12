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
## 5. 用变量a给出下面的定义

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
## 6.已知数组table，用宏求数组元素个数
```c
#define COUNT(table) (sizeof(table) / sizeof(table[0]))
```
## 7.带参宏和函数的区别？
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
## 8.内联函数的优缺点和适用场景是什么？
- 优点：内联函数与宏定义一样会在原地展开，省去了函数调用开销，同时又能做类型检查
- 缺点：它会使程序的代码量增大，消耗更多内存空间
## 9.如何用C语言实现读写寄存器变量？
```c
#define rBANKCON0 (*(volatile unsigned long *)0x48000004)
rBankCON0 = 0x12;
```
- 由于是寄存器地址，所以需要先将其强制类型转换为 volatile unsigned long *
## 10.下面代码能不能编译通过？
```c
#define c 3
c++;
```
- 不能。c宏定义是常数，++用于变量
## 11.在C语言中，凡是以#开头的都是预处理命令，同时预处理命令都是以#开头的

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