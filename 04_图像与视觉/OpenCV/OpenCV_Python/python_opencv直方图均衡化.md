# 实现函数
```python
equalized = cv2.equalizeHist(img)
```
# Code
```python
import cv2
import os

def equalize_gray_image(input_path, output_path=None):
    """
    对灰度图像进行直方图均衡化以增强对比度。
    
    参数:
        input_path: 输入灰度图路径（或彩色图，会自动转灰度）
        output_path: 输出图像路径（可选）
                     若不指定，则默认在同目录生成 *_equalized.jpg
                     
    返回:
        输出图像保存路径
    """
    # 读取图像
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")

    # 直方图均衡化
    equalized = cv2.equalizeHist(img)

    # 自动生成输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_equalized{ext}"

    # 保存结果
    cv2.imwrite(output_path, equalized)
    return output_path


# ===== 示例调用 =====
if __name__ == "__main__":
    result = equalize_gray_image("input_gray.jpg")
    print(f"均衡化后的图像已保存：{result}")

```