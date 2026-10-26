# 1.[[RegAD源码推理]]
# 补充知识
* [[AUC 简介]]

## 推理流程（三步）

1. **特征/图像对齐**：把查询图与支持集模板做配准（RegAD 用预训练特征 + 光流/形变场估计），消除位置与形变差异。
2. **重建/比对**：在对齐后的空间里重建正常样本（或把查询特征映射到正常特征空间），得到重建图/特征。
3. **残差打分**：`|query - recon|` 逐像素求差，平滑后取最大/分位数作为图像级异常分数，差值图即缺陷热力图。

```python
# 伪代码：推理主循环
support = torch.load("support.pt")             # 正常模板特征
for img in query_loader:
    feat = backbone(img)                        # 特征
    aligned = register(feat, support)           # 配准
    recon = reconstruct(aligned)                # 重建
    residual = (aligned - recon).abs().mean(1)  # 残差图
    score_map = gaussian_blur(residual)         # 平滑
    image_score = score_map.flatten().topk(k).values.mean()
    pred = score_map > threshold                # 像素级缺陷
```

## 关键实现细节

| 环节 | 注意 |
| --- | --- |
| 配准 | 形变太大会残留伪差异，导致误报；先做粗配准（模板匹配）再精配准 |
| 重建 | 重建能力太强会把缺陷也重建出来（漏检），要限制容量 |
| 打分 | 用 top-k 均值比 max 稳；阈值按正常样本的分位数标定 |
| 后处理 | 形态学开运算去噪点、连通域过滤小面积 |

## 代码与框图

- 源码级推理说明：[[RegAD源码推理]]
- 程序框图：[[推理程序框图]]
- 支持集为什么是 `.pt`：[[支持集是一个pt文件用来干什么的？]]

## 相关

- [[PatchCore]]、[[EfficientAd模型]] — 同类方法横向对比
- [[AUC 简介]] — 评价指标
- [[Mvtec数据组件]] — 数据加载
