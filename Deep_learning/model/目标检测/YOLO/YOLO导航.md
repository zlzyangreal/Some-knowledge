YOLO（You Only Look Once）是一种目标检测算法，它通过单次前向传递将对象检测和位置回归合并为一个端到端的模型

| Model (Year)   | 关键的建筑创新与贡献                                                                                                                             | Tasks                  | Framework             |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | --------------------- |
| YOLOv1 (2015)  | 第一台统一单级目标探测器(一个包围盒+类别概率网络)                                                                                                             | 目标检测、分类                | Darknet               |
| YOLOv2 (2016)  | 引入了多尺度训练；改进先验框( YOLO9000联合检测/分类)的锚点盒维数聚类                                                                                               | 目标检测、分类                | Darknet               |
| YOLOv3 (2018)  | Deeper Darknet-53 主干，有残余连接；加入SPP模块和多尺度特征融合进行小目标检测。                                                                                     | 对象检测，多尺度检测             | Darknet               |
| YOLOv4 (2020)  | 采用Mish激活函数；CSPDarknet - 53主干(跨阶段部分网络)，用于增强特征重用                                                                                         | 目标检测，目标跟踪              | Darknet               |
| YOLOv5 (2020)  | Ultralytics使用PyTorch实现；无锚检测头选项；使用SiLU ( Swish )激活和PANet颈进行特征聚合                                                                         | 对象检测，实例分割(限定)          | PyTorch (Ultralytics) |
| YOLOv6 (2022)  | 嵌入自注意力的高效Rep主干；为了提高效率，引入了无锚点目标检测模式                                                                                                     | 目标检测，实例分割              | PyTorch               |
| YOLOv7 (2022)  | 带模型重参数化的扩展ELAN ( E-ELAN )骨干网络；集成基于变压器的模块用于更广泛的任务( e.g.跟踪)                                                                              | 目标检测，目标跟踪，实例分割         | PyTorch               |
| YOLOv8 (2023)  | Ultralytics Next - Gen模型；新的C2f骨干网和去耦头；融入了生成式技术(基于GAN的增强)和完全无锚点设计                                                                       | 目标检测、实例分割、全景分割、关键点估计   | PyTorch (Ultralytics) |
| YOLOv10 (2024) | 通过一致的双分配训练策略(去除后处理)实现了端到端的NMS - free检测                                                                                                 | 目标检测                   | PyTorch               |
| YOLO11 (2024)  | 为了提高效率，在整个骨干网/颈部添加了C3k2 CSP瓶颈(更小的内核CSP块)；保留SPPF，引入C2PSA (具有空间注意力的CSP)模块，重点关注重要区域。将YOLO扩展到姿态估计和定向目标检测任务                                 | 目标检测、实例分割、位姿估计、定向检测    | PyTorch (Ultralytics) |
| YOLOv12 (2025) | 注意力中心架构：引入高效的区域注意力模块(低复杂度的全局自注意力机制)和残差ELAN ( Residual ELAN，R-ELAN )块改进特征聚合，以YOLO速度实现变压器级精度                                             | 目标检测                   | PyTorch               |
| YOLOv13 (2025) | 基于超图的自适应相关增强( HyperACE )模块，捕捉全局高阶特征交互；用于增强整个网络的特征流的全管道聚合分布( Full PAD )方案；利用深度可分离卷积来降低复杂度                                               | 目标检测                   | PyTorch               |
| YOLOv26 (2025) | Ultralytics边缘优化模型：使用原生的端到端预测器消除NMS；去掉DFL (分布聚焦损失)，使推理更简单、快速；引入Mu SGD优化器( SGD + Muon混合)，稳定快速收敛；显著提高了小目标的准确性，并且在低功耗设备上部署时，CPU推理速度提高了43 % | 目标检测、实例分割、姿态估计、定向检测、分类 | PyTorch (Ultralytics) |

1. [YOLOv1知识点](YOLOv1.md)
	-  [YOLOv1代码解析](YOLOv1代码.md)
2. [YOLOv2知识点](YOLOv2.md)
3. [YOLOv3知识点](YOLOv3.md)
4. [YOLOv4知识点](YOLOv4.md)
5. [YOLOv5知识点](YOLOv5.md)
	-  [YOLOv5代码解析及其操作](YOLOv5工程.md)
6. [[Yolov8知识点]]
	-  [YOLOv8资料](YOLOv8.md)
	- [[yolov8操作]]
7.  [[Yolo26知识点]]