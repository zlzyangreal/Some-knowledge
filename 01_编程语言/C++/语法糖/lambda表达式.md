能够捕获作用域中变量的匿名函数对象。因为C++是不能嵌套定义函数的，所以lambda就成了我们构造闭包的主要手段，不过在对象的生命周期上还是有点不同。

# 1.简述lambda
![[Lambda.png]]
## 1.1.捕获列表 `[]`
捕获列表是零或多个捕获符的逗号分隔符列表，可选地以默认捕获符开始（仅有的默认捕获符是 `&` 和 `=` ）。默认情况下，从lambda生成的类都包含一个对应该lambda所捕获变量的数据成员。类似任何普通类地数据成员，lambda的数据成员也在lambda对象创建时被初始化。类似参数传递，变量的捕获方式也可以是值或引用。
**值捕获：**
```cpp
void func()
{
	int i = 100;//局部变量
	//将i拷贝到明位f的可调用对象
	auto f = [i] { return i; };
	i = 0;
	int j = f(); //j=100,因为i是创建时拷贝的
}
```
**引用捕获：**
```cpp
void func()
{
	int i = 100;//局部变量
	//对象f包含i的引用
	auto f = [&i] { return i; };
	i = 0;
	int j = f(); //j=0,传递的是引用
}
```
**lambda捕获列表：**
```txt
[ ]。空捕获列表，lambda不能使用所在函数中的变量。
[=]。函数体内可以使用Lambda所在作用范围内所有可见的局部变量（包括Lambda所在类的this），并且是值传递方式（相当于编译器自动为我们按值传递了所有局部变量）。
[&]。函数体内可以使用Lambda所在作用范围内所有可见的局部变量（包括Lambda所在类的this），并且是引用传递方式（相当于编译器自动为我们按引用传递了所有局部变量）。
[this]。函数体内可以使用Lambda所在类中的成员变量。
[a]。将a按值进行传递。按值进行传递时，函数体内不能修改传递进来的a的拷贝，因为默认情况下函数是const的。要修改传递进来的a的拷贝，可以添加mutable修饰符。
[&a]。将a按引用进行传递。
[=，&a, &b]。除a和b按引用进行传递外，其他参数都按值进行传递。
[&, a, b]。除a和b按值进行传递外，其他参数都按引用进行传递。
```
## 1.2.形参列表 `( ) `
lambda形参列表和一般的函数形参列表类似，**但不允许默认实参(C++14 前)**。当以 auto 为形参类型时，该 lambda 为泛型 lambda(C++14 起)。与一个普通函数调用类似，调用一个lambda时给定的实参被用来初始化lambda的形参。
```cpp
//代码在VS2019中测试
void func()
{
	int i = 1, j = 2;
	auto f = [](int a,int &b) {
		a = 10;
		b = 20;
		//输出：10  20
		std::cout << a << " " << b << std::endl;
	};
	f(i,j);
	//输出：1 20
	std::cout << i << " " << j << std::endl;
}
```
```cpp
//代码在VS2019中测试
void func()
{
	auto f = [](auto a,int b=10) {
		std::cout << a << " " << b << std::endl;
	};
	// 输出1.5 2
	f(1.5, 2);
	// 输出1 10
	f(true);
}
```
##  1.3.说明符
```txt
mutable：允许 函数体 修改各个复制捕获的对象，以及调用其非 const 成员函数；
constexpr：显式指定函数调用运算符为 constexpr 函数。此说明符不存在时，若函数调用运算符恰好满足针对 constexpr 函数的所有要求，则它也会是 constexpr；  (C++17 起)
consteval：指定函数调用运算符为立即函数。不能同时使用 consteval 和 constexpr。(C++20 起)
```
默认情况下，对于一个值被拷贝的变量，lambda不会改变其值。假如我们希望能改变一个被捕获的变量的值，就必须在参数列表后面加上关键字mutable。而一个引用捕获的变量则不受此限制。 
```cpp
//代码在VS2019中测试
void func()
{
	int i = 10, j = 10;
	//加上mutable才可以在lambda函数中改变捕获的变量值
	auto f = [i, &j]() mutable {
		i = 100, j = 100;
	};
	i = 0, j = 0;
	f();
	//输出:0 100
	std::cout << i << " " << j << std::endl;
}
```
## 1.4.返回类型` ->`
当我们需要为一个lambda定义返回类型时，需要使用尾置返回类型。返回类型若缺省，则根据函数体中的 return 语句进行推断（如果有多条return语句，需要保证类型一直，否则编译器无法自动推断）。默认情况下，如果一个lambda函数体不包含return语句，则编译器假定返回void。
```cpp
void func()
{
	auto f = []() ->double {
		if (1 > 2)
			return 1;
		else 
			return 2.0;
	};
	std::cout << f() << std::endl;
}
```
如果不显示指定返回类型，则int和double两种返回类型会导致推断冲突。
![[erro.png]]
## 1.5 函数体 `{ }`
略，同普通函数的函数体。
# 2.eg
```cpp
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>
 
int main()
{
    std::vector<int> c = { 1, 2, 3, 4, 5, 6, 7 };
    int x = 5;
    c.erase(std::remove_if(c.begin(), c.end(), [x](int n) { return n < x; }), c.end());
 
    std::cout << "c: ";
    std::for_each(c.begin(), c.end(), [](int i) { std::cout << i << ' '; });
    std::cout << '\n';
 
    // 闭包的类型不能被指名，但可用 auto 提及
    // C++14 起，lambda 能拥有自身的默认实参
    auto func1 = [](int i = 6) { return i + 4; };
    std::cout << "func1: " << func1() << '\n';
 
    // 与所有可调用对象相同，闭包能可以被捕获到 std::function 之中
    // （这可能带来不必要的开销）
    std::function<int(int)> func2 = [](int i) { return i + 4; };
    std::cout << "func2: " << func2(6) << '\n';
 
    system("pause");
    return 0;
}
```
![[Lambda输出.png]]