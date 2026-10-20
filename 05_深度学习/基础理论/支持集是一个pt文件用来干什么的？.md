# 支持集是一个pt文件用来干什么的？

> “支持集（support set）存成 .pt”是**小样本/无监督异常检测**里的常见做法：把若干张“正常样本”的
> 特征（或整张图/掩膜）提前算好保存下来，推理时只算查询样本与它的距离，从而判断“像不像正常”。

## 1. support set 是什么

在 few-shot / 度量学习里，数据被分成：

| 名称 | 含义 | 例子 |
| --- | --- | --- |
| support set（支持集） | 已知类别的少量样本，用来“定义类别” | 每类 1~5 张正常图 |
| query set（查询集） | 要判定的新样本 | 待检测的工件图 |

模型不训练分类头，而是学一个**嵌入空间**：query 离哪一类 support 最近，就判为哪一类；
离所有 support 都远 → 判为异常/新类。

## 2. 为什么存成 .pt

`.pt` 是 PyTorch 的序列化格式（`torch.save/load`），存的是**张量/字典**，不是模型结构：

```python
# 离线：为每个产品/每个工位预先算好支持集特征
import torch
feats = torch.stack([extract_feature(img) for img in normal_imgs])   # [K, D]
torch.save({"features": feats, "mean": feats.mean(0), "std": feats.std(0)},
           "bottle_support.pt")

# 在线：推理只加载特征，不跑 backbone 训练
support = torch.load("bottle_support.pt")
d = torch.cdist(query_feat, support["features"])      # 到每个支持样本的距离
score = d.min(dim=1).values                           # 最近邻距离 = 异常分数
```

好处：

1. **省算力**：支持集特征只算一次，推理端可以只跑一次前向。
2. **免训练**：新产品来了不用重训网络，重新采几张正常图算特征即可（PatchCore 的核心思想）。
3. **可复现**：特征固定，阈值调参不受随机性影响。

## 3. 典型使用场景

- **PatchCore / SPADE 类异常检测**：支持集 = 正常图的 patch 特征库（memory bank）。
- **小样本缺陷检测**：支持集 = 每种缺陷的少量标注样本。
- **RegAD**：支持集 = 配准用的正常模板特征，见 [[RegAD 推理]]。
- **分割/检测的 prototype**：支持集 = 每类原型向量。

## 4. 用的时候注意

1. **支持集要与查询集同分布**：同一个相机、同一光照、同一工位，否则距离全错。
2. **特征要归一化**：L2 归一化后再算余弦/欧氏距离，否则被幅值主导。
3. **支持集大小**：太少（1~2 张）噪声大，太多则推理变慢；PatchCore 用 coreset 采样压到几百个 patch。
4. **阈值要标定**：用一批正常图算分数分布，取 99% 分位数之类作为阈值，别拍脑袋。

## 相关

- [[RegAD 推理]]、[[PatchCore]]、[[无监督异常检测索引]]
- [[EfficientAd模型]]、[[DINO资料]]
- [[AUC 简介]] — 评价这套打分是否靠谱
