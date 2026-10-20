```python
import cv2
import numpy as np
import os

def morphological_close(input_path, output_dir,
                        kernel_size=(5,5),
                        kernel_shape='ellipse',
                        iterations=1,
                        output_name="edges_closed.png"):
    """
    对二值边缘图执行形态学闭运算 (closing)，用于连接断裂的边缘。

    参数:
      input_path (str): 输入图像路径（建议是 Canny 输出的二值图）。
      output_dir (str): 输出目录（会自动创建）。
      kernel_size (tuple): 结构元大小 (width, height)，通常为奇数。
      kernel_shape (str): 结构元形状：'rect'、'ellipse'、'cross'。
      iterations (int): 形态学操作迭代次数（>1 可增强连接，但可能使边变粗）。
      output_name (str): 保存的文件名。

    返回:
      closed (ndarray): 闭运算后的二值图像（uint8）。
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {input_path}")

    # 如果输入是彩色或灰度图，转换成单通道二值（如果已经是二值则保持）
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 如果不是二值图（0/255），就二值化一下（自动阈值）
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

    # 闭运算（先膨胀后腐蚀）
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    out_path = os.path.join(output_dir, output_name)
    cv2.imwrite(out_path, closed)
    print(f"[INFO] 闭运算结果已保存到: {out_path}")

    return closed

# ========== 示例用法 ==========
if __name__ == "__main__":
    input_edges = "/mnt/data/edges_canny_smooth.png"   # 你已生成的 Canny 图
    output_folder = "./morph_output"

    # 常用参数建议：
    # kernel_size=(3,3) -> 连接近邻的小断裂；(5,5) 或 (7,7) -> 连接更大的断裂，但会把细边变粗
    # kernel_shape='ellipse' 通常效果自然
    closed_img = morphological_close(input_edges,
                                     output_folder,
                                     kernel_size=(5,5),
                                     kernel_shape='ellipse',
                                     iterations=1,
                                     output_name="edges_closed.png")

```
- `kernel_size`：小（3×3）只连接很短的断裂；中（5×5）适用于多数场景；大（7×7+）会把边缘变宽，慎用。
- `kernel_shape='ellipse'` 对自然/曲线边缘通常效果最好；`rect` 对直线更强；`cross` 更细节化。
- `iterations`：增加迭代次数可以加强连接，但会让边更粗。通常 `1` 就足够。