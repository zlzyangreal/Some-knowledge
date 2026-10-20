# 可变数据类型和不可变数据类型
![[数据类型.png]]

![[可变类型变量引用.png]]

![[可变类型变量引用2.png]]

## 1. 哪些可变、哪些不可变

| 不可变（immutable） | 可变（mutable） |
| --- | --- |
| int、float、bool、complex | list |
| str | dict |
| tuple（内部元素不可变） | set |
| frozenset | 自定义类实例（默认可变） |
| bytes | bytearray |

## 2. “不可变”到底是什么意思

不可变的含义是：**对象创建后，它的值不能被原地修改**。看起来的“修改”其实是新建了一个对象并把名字重新绑过去。

```python
a = "abc"
b = a
a += "d"        # 等价于 a = a + "d"，新建 str 对象
print(a, b)     # abcd abc  —— b 不受影响
```

```python
x = [1, 2]
y = x
x.append(3)     # 原地修改
print(x, y)     # [1, 2, 3] [1, 2, 3]  —— 同一个对象
```

判断是否同一对象用 `is`（比 id），判断值相等用 `==`。

## 3. 为什么可变/不可变很重要

1. **可哈希**：只有不可变对象才能当 dict 的键、放进 set。`[1,2]` 放 set 会 `TypeError: unhashable type`。
2. **函数默认参数陷阱**：

```python
def add(x, bucket=[]):      # 错：默认值只创建一次，被所有调用共享
    bucket.append(x)
    return bucket

def add_ok(x, bucket=None): # 对：用 None 占位，函数内再建
    if bucket is None:
        bucket = []
    bucket.append(x)
    return bucket
```

3. **浅拷贝 / 深拷贝**：

```python
import copy
a = [[1, 2], [3, 4]]
b = a.copy()          # 浅拷贝：内层 list 仍是同一对象
c = copy.deepcopy(a)  # 深拷贝：递归复制
a[0][0] = 99
print(b[0][0])        # 99 —— 被牵连
print(c[0][0])        # 1
```

4. **tuple 里放 list**：tuple 本身不可变，但里面的 list 内容可以改。

## 图示对应

上面三张图分别是：数据类型全貌、`b = a` 时两个名字指向同一对象的示意、以及列表原地修改后两个名字都“看到”新值的示意。

## 相关

- [[python基本类型]]、[[python容器类型]]
- [[python的可迭代对象、迭代器与生成器]]
- [[拷贝构造函数]]（C++ 里同名概念的对照）
