**类型自动推断**

用法包括两种：一是在变量声明时根据初始化表达式自动推断该变量的类型，二是在声明函数时作为函数返回值的占位符

自动类型推断
--------------
例子：对于vector类型的变量，如果我们需要获取它的迭代器，我们需要这样声明``vector::iterator iter``，而使用auto关键字后我们可以让编译器帮我们推断出迭代器的具体类型。另外，在模板函数定义时，如果变量的类型依赖于模板参数，我们也很难确定变量的类型，使用auto关键字则可以把这些“脏活累活”交给编译器完成
```c++
#include <iostream> 
#include <vector> 
using namespace std;

template<class T, class U>
void add(T t, U u)
{
    auto s = t + u;
    cout << "type of t + u is " << typeid(s).name() << endl;
}

int main()
{
    // 简单自动类型推断
    auto a = 123;
    cout << "type of a is " << typeid(a).name() << endl;
    auto s("fred");
    cout << "type of s is " << typeid(s).name() << endl;

    // 冗长的类型说明（如迭代器）
    vector<int> vec;
    auto iter = vec.begin();
    cout << "type of iter is " << typeid(iter).name() << endl;

    // 使用模板技术时，如果某个变量的类型依赖于模板参数，使用auto确定变量类型
    add(101, 1.1);
}
```

函数返回值占位符
---------------
auto主要与decltype关键字配合使用，作为返回值类型后置时的占位符。此时，关键字不表示自动类型检测，仅仅是表示后置返回值的语法的一部分
```c++
template<class T, class U>
auto add(T t, U u) -> decltype(t + u) 
{
    return t + u;
}
```

注意点
------
1   使用auto关键字的变量必须有初始值

2   可以使用valatile，`*`（指针类型说明符），`&`（引用类型说明符），`&&`（右值引用）来修饰auto关键字
```c++
auto a = 10;
auto *pa = new auto(a);
auto **rpa = new auto(&a);
cout << typeid(a).name() << endl;   // 输出： int
cout << typeid(pa).name() << endl;  // 输出： int *
cout << typeid(rpa).name() << endl; // 输出： int **
```
3   函数参数和模板参数不能被声明为auto

4   使用auto关键字声明变量的类型，不能自动推导出顶层的CV-qualifiers和引用类型，除非显示声明
* 使用auto关键字进行类型推导时，如果初始化表达式是引用类型，编译器会去除引用，除非显示声明
```c++
int i = 10;
int &r = i;
auto a = r;
a = 13; // 重新赋值
cout << "i = " << i << " a = " << a << endl;    // 输出i=10，a=13

// 显式声明
auto &b = r;
b = 15; // 重新赋值
cout << "i = " << i << " b = " << b << endl;    // 输出i=15，a=15
```
* 使用auto使用auto关键字进行类型推导时，编译器会自动忽略顶层const，除非显示声明
```c++
const int c1 = 10;
auto c2 = c1;
c1 = 11; // 报错，c1为const int类型，无法修改const变量
c2 = 14; // 正确，c2为int类型

// 显示声明
const auto c3 = c1;
c3 = 15; // 报错，c3为const int类型，无法修改const变量
```
* 对于数组类型，auto关键字会推导为指针类型，除非被声明为引用
```c++
int a[10];
auto b = a;
cout << typeid(b).name() << endl;   // 输出：int *

auto &c = a;
cout << typeid(c).name() << endl;   // 输出：int [10]
```

