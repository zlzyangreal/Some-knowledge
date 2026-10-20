库源码：
* https://github.com/tensorflow/lingvo 
* https://github.com/qubvel/efficientnet

## 当时的背景

GPipe 最早是 Google 2019 年的论文（Huang et al.），官方实现基于 **TensorFlow 1.x + Lingvo**，
用 `tf.contrib` 的流水线 API 把模型按层切到多 TPU/GPU。现在这套 API 在 TF2 里已不存在。

## 现在在 TF 里怎么做流水线并行

| 方案 | 说明 |
| --- | --- |
| `tf.distribute.MirroredStrategy` | 单机多卡的**数据并行**，最常用，见 [[单机多卡训练： MirroredStrategy]] |
| `tf.distribute.MultiWorkerMirroredStrategy` | 多机多卡数据并行 |
| `tf.distribute.TPUStrategy` | TPU 上由 XLA 自动做模型并行 |
| 手动 `tf.device('/GPU:0')` + 分段 `tf.function` | 手写流水线，灵活但麻烦 |

TF2 的官方态度是：**优先数据并行**；模型大到一张卡放不下时，转 PyTorch + 流水线，或用 JAX 的 `pjit`。
所以这条线上的实践基本已经迁走了。

## MirroredStrategy 最小示例（现实中最常用的替代）

```python
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()     # 自动发现所有 GPU
print("replicas:", strategy.num_replicas_in_sync)

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(224, 224, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

model.fit(train_ds, epochs=10, validation_data=val_ds)
```

注意：`model` 必须在 `strategy.scope()` **里面**创建，否则变量不会被复制到各卡。

## 大模型时代的现实建议

1. TF 生态现在主要维护推理侧（TFLite/TF Serving），训练侧新特性少。
2. 要做流水线/张量并行，直接看 PyTorch 的 `torch.distributed.pipelining`，见 [[GPipe(Pytorch版本)]]。
3. 只想多卡加速训练 → `MirroredStrategy`（TF）或 DDP（PyTorch），别一上来就流水线。

## 相关

- [[GPipe(Pytorch版本)]]
- [[单机多卡训练： MirroredStrategy]]
- [[tensorflows Lite是什么]]、[[tensorflows系列知识]]
- 参考：[Lingvo](https://github.com/tensorflow/lingvo)
