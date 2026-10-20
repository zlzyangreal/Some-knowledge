# 1.[[Numpy]]
# 2.[[Pandas]]
# 3.[[OpenCV简介|OpenCv]]

## 各自负责什么

| 库 | 定位 | 典型用途 |
| --- | --- | --- |
| NumPy | 多维数组与向量化运算 | 像素级运算、掩膜、通道操作 |
| Pandas | 表格数据 | 读 csv 标注、统计分析、导出结果 |
| OpenCV (cv2) | 工业级图像处理 | 读写、滤波、形态学、检测、视频流 |
| Pillow (PIL) | 轻量图像读写与简单处理 | 格式转换、缩放、加水印、字体绘制 |
| scikit-image | 算法教学/科研 | 分水岭、骨架化、HOG、SSIM 等指标 |
| Matplotlib | 可视化 | 显示图像、画直方图、画检测框 |

## NumPy 与图像的关系

图像在 NumPy 里就是 `ndarray`：

```python
import numpy as np, cv2

img = cv2.imread("a.jpg")              # (H, W, 3)，BGR 顺序！不是 RGB
print(img.shape, img.dtype)            # (480, 640, 3) uint8

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
roi  = img[100:200, 200:300]           # 切片即裁剪
flip = img[::-1]                       # 上下翻转
mean = img.mean(axis=(0, 1))           # 每个通道均值

# 掩膜运算
mask = (gray > 128).astype(np.uint8)
result = cv2.bitwise_and(gray, gray, mask=mask)
```

要点：

- **BGR vs RGB**：OpenCV 读进来是 BGR，用 matplotlib 显示前要 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`，否则红蓝反转。
- **数据类型**：`uint8` 运算会溢出回绕（255+1=0），相减/相加要用 `cv2.add/subtract` 或先转 `float32`。
- **连续内存**：切片得到的数组可能不连续，喂给某些库前 `np.ascontiguousarray()`。

## Pandas 处理标注文件

```python
import pandas as pd

df = pd.read_csv("labels.csv")                 # x1,y1,x2,y2,cls
stats = df.groupby("cls").size().sort_values(ascending=False)
df["w"] = df.x2 - df.x1
df["h"] = df.y2 - df.y1
df["area"] = df.w * df.h
df.to_csv("labels_stat.csv", index=False)
```

## 常见组合

```python
# 读 → 处理 → 存
img = cv2.imread("in.png", cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
cv2.imwrite("out.png", edges)
```

## 相关

- [[Numpy]]、[[Pandas]]、[[Pandas原理]]
- [[OpenCV简介]]、[[OpenCV_Python索引]]
- [[科学计算库索引]]
