![[python_opencv拉普拉斯锐化.png]]
# Code
```python
import cv2
import numpy as np
import os

def laplacian_sharpen_gray(input_path, output_path=None, alpha=1.0):
    """
    对灰度图像进行拉普拉斯锐化增强。
    
    参数:
        input_path: 输入图像路径（彩色或灰度）
        output_path: 输出图像路径（可选，默认为 *_lapsharp.jpg）
        alpha: 锐化系数（通常 0.5 ~ 2.0，值越大锐化越强）
    
    返回:
        输出图像保存路径
    """
    # 读取图像（灰度）
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")

    # 计算拉普拉斯（使用 3x3 核）
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

    # 组合锐化结果：原图 - α * Laplacian
    sharp = cv2.convertScaleAbs(img - alpha * lap)

    # 自动生成输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_lapsharp{ext}"

    # 保存结果
    cv2.imwrite(output_path, sharp)
    return output_path


# ===== 示例调用 =====
if __name__ == "__main__":
    result = laplacian_sharpen_gray("input_gray.jpg", alpha=1.2)
    print(f"拉普拉斯锐化图已保存：{result}")

```
![[python_opencv拉普拉斯锐化1.png]]
# 算法效果
![[python_opencv拉普拉斯锐化2.png]]