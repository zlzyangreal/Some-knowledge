# new and delete
c++中构造heap对象
```c++
Complex* pc = new Comple(1,2);
```
编译器执行操作：
```c++
void* men = operator new(sizeof(Complex));  //分配内存
pc = static_cast<Complex*>(men);             //转型
pc -> Complex::Complex(1,2);                //构造函数
```
c++的释放内存
```c++
Complex* ps = new Comple;
...
delete ps;
```
编译器操作：
```c++
String::~String(ps);  //析构函数
operator delete(ps);  //释放内存
```
数组释放：
```c++
String* p = new String[3];
...
delete[] pc;
```
否则会造成内存泄漏，原因是可能造成释放内存唤起次数不够
**注意**:
* 不要使用`delete`来释放不是`new`分配的内存。
* 不要使用`delete`释放同一个内存块两次。
* 如果使用`new [ ]`为数组分配内存，则应使用`delete [ ]`来释放。
* 如果使用`new [ ]`为一个实体分配内存，则应使用`delete`（没有方括号）来释放。
* 对空指针应用`delete`是安全的。