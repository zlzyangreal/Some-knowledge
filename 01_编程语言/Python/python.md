# 1.基础
## 1.1 [[python if while for]]
## 1.2 [[python基本类型]]
## 1.3 [[python输入输出]]
## 1..4 [[python的可迭代对象、迭代器与生成器]]
## 1.5 [[python可变数据类型和不可变数据类型]]
## 1.6 [[python容器类型]]
# 2.各种函数与库
## 2.1 关键字
### 2.1.1[[关键字with]]
### 2.1.2 [[关键字raise]]
## 2.2 函数
### 2.2.1 [[函数isupper()]]
### 2.2.2 [[f-string 用法]]
## 2.3 库
### 2.3.1 [[os 库]]
### 2.3.2 [[torch 库]]

## 3. 学习路线（按这个顺序看本目录）

1. **语法层**：[[python基本类型]] → [[python容器类型]] → [[python if while for]] → [[python输入输出]]
2. **数据模型**：[[python可变数据类型和不可变数据类型]] → [[python的可迭代对象、迭代器与生成器]]
3. **工程习惯**：[[关键字with]]（文件/锁自动释放）、[[关键字raise]]（抛异常）、[[f-string 用法]]
4. **常用库**：[[os 库]] → [[argparse模块]] → [[torch 库]]

## 4. 一页速查

```python
# 变量与类型（动态类型，但强类型）
x = 1                # int
s = "abc"            # str，不可变
lst = [1, 2, 3]      # list，可变
tup = (1, 2)         # tuple，不可变
d = {"a": 1}         # dict，键必须是可 hash 的对象
st = {1, 2}          # set，自动去重

# 切片 [start:stop:step]，左闭右开
lst[::-1]            # 反转
lst[1:3]             # 第 2、3 个元素

# 推导式
squares = [i * i for i in range(10) if i % 2 == 0]
mapping = {k: v for k, v in zip("abc", range(3))}

# 解包
a, b = b, a          # 交换
first, *rest = [1, 2, 3]

# 常用内建
enumerate(lst), zip(a, b), sorted(lst, key=len, reverse=True)
any(x > 0 for x in lst), all(x > 0 for x in lst)
```

## 5. 和 C++ 的区别（最容易踩的 5 条）

1. 变量是**名字绑定到对象**，没有声明类型；赋值只是换绑定。
2. 函数参数是“对象引用传递”，但重新赋值不影响调用者（因为只是重新绑定名字）。
3. 没有块级作用域，只有函数/模块/类作用域。
4. `for` 是迭代器协议，不是 C 风格三段式；`range()` 返回惰性序列。
5. 一切皆对象，函数也是对象，可以当参数传（这是后面 [[f-string 用法]]、闭包、装饰器的基础）。

## 相关

- [[主函数模块区分条件]]（`if __name__ == "__main__"`）
- [[python深度学习]]、[[C-python用法|C 与 Python 互调]]
