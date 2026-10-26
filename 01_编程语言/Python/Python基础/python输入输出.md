# python 输入输出
![[输入输出.png]]

## 输入

```python
name = input("请输入姓名：")      # 返回值永远是 str
age = int(input("年龄："))         # 要数字就自己转
nums = list(map(int, input().split()))   # 一行多个数字
```

## 输出

```python
print("a", "b", sep="-", end="\n")   # 默认 sep=' ' end='\n'
print(f"{name} 今年 {age} 岁")        # f-string，最常用
print("{} 岁".format(age))
print("%d 岁" % age)                  # 老式，读老代码会遇到
```

格式化细节（小数位、对齐、千分位）见 [[f-string 用法]]。

## 文件读写

```python
# 文本，推荐 with：出异常也会自动 close
with open("a.txt", "r", encoding="utf-8") as f:
    for line in f:            # 逐行迭代，大文件也不炸内存
        print(line.rstrip())

with open("b.txt", "w", encoding="utf-8") as f:
    f.write("hello\n")

# 一次性读
text = open("a.txt", encoding="utf-8").read()
lines = open("a.txt", encoding="utf-8").readlines()
```

模式速记：

- `r` 读、`w` 覆盖写、`a` 追加、`x` 独占创建（存在就报错）
- 加 `b` 是二进制（图片、模型文件、音频必须用 `rb/wb`）
- 文本模式务必写 `encoding="utf-8"`，Windows 默认是 GBK，中文会乱码

## 补充图片说明

上图是 `print` 的几种输出形式与 `input` 的返回值类型；重点记住 **`input()` 返回 str**，参与算术前必须转换。

## 相关

- [[关键字with]]
- [[python基本类型]]、[[python容器类型]]
- [[f-string 用法]]
