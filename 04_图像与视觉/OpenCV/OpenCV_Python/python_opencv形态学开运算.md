```python
import cv2
import numpy as np
import os

def morphological_open(input_path, output_dir,
                       kernel_size=(3,3),
                       kernel_shape='ellipse',
                       iterations=1,
                       output_name="edges_opened.png"):
    """
    对二值边缘图执行形态学开运算 (opening)，用于去除孤立噪点。

    参数:
      input_path (str): 输入图像路径（建议是 Canny 或闭运算输出图）。
      output_dir (str): 输出目录（会自动创建）。
      kernel_size (tuple): 结构元大小 (width, height)，通常为奇数。
      kernel_shape (str): 结构元形状：'rect'、'ellipse'、'cross'。
      iterations (int): 形态学操作迭代次数。
      output_name (str): 输出文件名。

    返回:
      opened (ndarray): 开运算后的二值图像（uint8）。
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {input_path}")

    # 如果输入是彩色或灰度图，转换为单通道
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 确保输入为二值图
    unique_vals = np.unique(gray)
    if not (set(unique_vals) <= {0,255}):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = gray

    # 选择结构元素形状
    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    if kernel_shape not in shape_map:
        raise ValueError("kernel_shape 必须是 'rect'、'ellipse' 或 'cross'。")

    kernel = cv2.getStructuringElement(shape_map[kernel_shape], kernel_size)

    # 开运算（先腐蚀后膨胀）
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)

    out_path = os.path.join(output_dir, output_name)
    cv2.imwrite(out_path, opened)
    print(f"[INFO] 开运算结果已保存到: {out_path}")

    return opened

# ========== 示例用法 ==========
if __name__ == "__main__":
    input_edges = "./morph_output/edges_closed.png"  # 你上一步闭运算的结果
    output_folder = "./morph_output"

    opened_img = morphological_open(input_edges,
                                    output_folder,
                                    kernel_size=(3,3),
                                    kernel_shape='ellipse',
                                    iterations=1,
                                    output_name="edges_opened.png")

```

| 参数             | 建议值               | 说明                     |
| -------------- | ----------------- | ---------------------- |
| `kernel_size`  | `(3,3)` 或 `(5,5)` | 小尺寸更保留细节，大尺寸更强去噪       |
| `kernel_shape` | `'ellipse'`       | 对圆形/管状噪点去除更自然          |
| `iterations`   | `1`               | 一般一次即可，>1 会去掉更多但可能损伤边缘 |