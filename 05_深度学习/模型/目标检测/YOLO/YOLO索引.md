# YOLO

YOLO 系列原理、代码与改进（含注意力、损失函数、数据格式转换）。

上级：[[目标检测索引]]　|　总索引：[[总索引]]

## 笔记（20）

- [[CA注意力机制]] — 包括单独的CA，以及将CA融入C3k2
- [[CBAM注意力机制]] — 和EMA注意力机制一样
- [[EMA注意力机制]] — ema.py（From the official source）
- [[SimAM注意力机制]] — ultralytics/ultralytics/nn/modules file_tree
- [[TransNeXt注意力机制]] （纯代码）
- [[VOC转COCO]] （纯代码）
- [[Wise-IoU损失函数]] — /home/alphafox/Desktop/Code/Yolo26/ultralytics/ultralytics/c
- [[YOLOv1]] — YOLO是one-stage
- [[YOLOv1代码]] — 按照一定的流程加载数据、构建模型、进行训练和评估，并保存训练好的模型参数
- [[YOLOv2]] — YOLOv2 相对于 YOLOv1 的提升
- [[YOLOv3]] — 论文译文 https://www.cnblogs.com/wj-1314/p/9744146.html
- [[YOLOv4]] — 博文 https://www.jiangdabai.com/2120
- [[YOLOv5]] — 博文 https://blog.csdn.net/qq_37541097/article/details/1235943
- [[YOLOv5工程]] — weights下加入权重文件 yolov5l/m/s/x.pt文件
- [[YOLOv8]] — https://github.com/ultralytics/ultralytics
- [[YOLO导航]] — YOLO（You Only Look Once）是一种目标检测算法，它通过单次前向传递将对象检测和位置回归合并为一个端到
- [[Yolo26知识点]] — Yolo26 专为边缘和低功耗设备而设计
- [[Yolov8知识点]] — Conv：基础卷积操作，用于特征初步抽象。
- [[Yolo改]] — SimAM注意力机制 — 无参注意力，按能量函数给每个位置权重，几乎不增参数量。
- [[yolov8操作]] — 使用Pip在一个Python=3.8环境中安装ultralytics包，此环境还需包含PyTorch=1.7。

## 附件文件（4，30.5 KB）

类型：.py×2、.yaml×2

- `VOC_TO_COCO_way版.py`
- `mydefault 1.yaml`
- `mydefault.yaml`
- `voc_to_coco.py`
