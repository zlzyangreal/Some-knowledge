`option` 结构体用于与 [[getopt_long]] 函数一起使用，以定义长选项的名称、参数需求等信息
```c
/* Names for the values of the 'has_arg' field of 'struct option'.  */
#define no_argument     0
#define required_argument  1
#define optional_argument  2


struct option
{
  const char *name;//长选项的名字
  /* has_arg can't be an enum because some compilers complain about
     type mismatches in all the code that assumes it is an int.  */
  int has_arg;    //选项是否需要参数
  int *flag;      //指向一个整型变量的指针，用于存储选项的状态
  int val;        //选项的值，通常是一个字符，对应短选项
};
```
* [[getopt_long|例子]]