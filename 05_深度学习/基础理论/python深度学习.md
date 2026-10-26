https://ai.renyuzhuo.cn/books/DeepLearningWithPython/CH-DeepLearningWithPython.pdf

## 这本书怎么读

《Python 深度学习》（Deep Learning with Python，François Chollet，Keras 作者）分两部分：

| 部分 | 内容 | 读法 |
| --- | --- | --- |
| 第一部分（1–4 章） | 什么是深度学习、张量、梯度、神经网络入门 | 必读，把概念打通 |
| 第二部分（5–9 章） | 计算机视觉、序列、文本、生成模型、最佳实践 | 按需读，配合项目 |

## 核心概念对照本库

- 张量 → [[张量基础知识]]
- 梯度与反向传播 → [[前向和后向传播]]
- 损失与优化 → [[神经网络中的Epoch、Iteration、Batchsize]]
- 卷积 → [[卷积块]]、[[网络主干概括]]
- 数据增强 → [[马赛克数据增强]]、[[数据增强索引]]
- 过拟合与正则 → [[推理时为什么设置随机数种子]]、[[AUC 简介]]

## 书里的代码现在要注意

书用 Keras（TensorFlow 2.x）。要注意：

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
```

- `tf.keras` 与独立 `keras` 包现在已合并，老代码里 `from keras...` 可能报版本错，见 [[Keras🍏 and tf.keras 🍎]]。
- 书里没有 PyTorch 版本，但概念完全对应，想转 PyTorch 看 [[PyTorch基础索引]]。

## 建议的练习路径

1. 第 2 章：手写一遍 MNIST 全连接，理解 `fit` 到底做了什么。
2. 第 5 章：猫狗分类，重点练数据增强与迁移学习。
3. 第 6 章：文本分类，理解 embedding 与序列。
4. 第 8 章：生成模型，配合 [[GAN是什么]]。

## 相关

- [[计算机视觉课程]]
- [原书 PDF](https://ai.renyuzhuo.cn/books/DeepLearningWithPython/CH-DeepLearningWithPython.pdf)
