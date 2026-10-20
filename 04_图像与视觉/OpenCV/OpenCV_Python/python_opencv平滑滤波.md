# Code
```python
import cv2
import os

def mild_smooth_gray(input_path, output_path=None, method="bilateral", strength=1):
    """
    对灰度图像进行轻微平滑，去除细微噪声和干扰，同时保留边缘。
    
    参数:
        input_path: 输入图像路径（灰度或彩色均可）
        output_path: 输出路径（可选，默认 *_smooth.jpg）
        method: 平滑方法，可选：
                - "bilateral"：双边滤波（推荐，保边去噪）
                - "gaussian"：高斯平滑
                - "average" ：均值滤波
        strength: 平滑强度（1=轻微，2=中等，3=较强）
    
    返回:
        输出图像保存路径
    """
    # 读取图像（灰度）
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")

    # 轻微平滑
    if method == "bilateral":
        # 双边滤波（保边）
        d = 3 + 2 * strength        # 邻域直径
        sigma_color = 30 * strength # 颜色空间平滑系数
        sigma_space = 15 * strength # 坐标空间平滑系数
        smooth = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    elif method == "gaussian":
        ksize = 3 + 2 * strength
        smooth = cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif method == "average":
        ksize = 3 + 2 * strength
        smooth = cv2.blur(img, (ksize, ksize))
    else:
        raise ValueError("method 参数必须是 'bilateral'、'gaussian' 或 'average'")

    # 输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_smooth{ext}"

    cv2.imwrite(output_path, smooth)
    return output_path


# ===== 示例调用 =====
if __name__ == "__main__":
    result = mild_smooth_gray("input_gray.jpg", method="bilateral", strength=1)
    print(f"轻微平滑结果已保存：{result}")

```
![[python_opencv平滑滤波.png]]