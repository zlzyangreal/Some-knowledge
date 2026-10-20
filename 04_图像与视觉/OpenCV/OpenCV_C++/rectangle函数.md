`rectangle`函数用于在图像上绘制矩形
# 1.原型
```cpp
cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]])
```
参数说明：
- `img`: 要在其上绘制矩形的图像。
- `pt1`: 矩形左上角的坐标点（以(x, y)的形式）。
- `pt2`: 矩形右下角的坐标点（以(x, y)的形式）。
- `color`: 矩形的颜色，可以是BGR格式的元组，例如(255, 0, 0)表示蓝色。
- `thickness`: 线条的粗细。如果设置为-1，则矩形将被填充。
- `lineType`: 线条的类型，例如`cv2.LINE_4`或`cv2.LINE_8`。
- `shift`: 点坐标和线条粗细的缩放因子。