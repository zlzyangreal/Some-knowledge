# 1.单一文件
```python
import cv2
import os

def convert_to_gray(input_path, output_path=None):
    """
    将输入图像转换为灰度图并保存。
    参数:
        input_path:  输入图像路径
        output_path: 输出图像路径（可选）
                      如果不提供，自动在同目录下生成 *_gray.jpg 文件
    返回:
        输出文件的保存路径
    """
    # 读取图像
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{input_path}")

    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 若未指定输出路径，则自动命名
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_gray{ext}"

    # 保存灰度图
    cv2.imwrite(output_path, gray)
    return output_path


# ===== 示例调用 =====
if __name__ == "__main__":
    result_path = convert_to_gray("input.jpg")
    print(f"灰度图已保存：{result_path}")

```
# 2. 文件夹内全部转灰度
```python
import cv2
import os
import glob

def batch_convert_to_gray(input_folder, output_folder):
    """
    批量将文件夹中的所有图片转换为灰度并保存到指定目录
    支持 jpg/png/jpeg/bmp/tif 等格式
    """
    os.makedirs(output_folder, exist_ok=True)

    # 支持多种图片格式
    img_formats = ("*.jpg", "*.png", "*.jpeg", "*.bmp", "*.tif")
    img_paths = []
    for fmt in img_formats:
        img_paths.extend(glob.glob(os.path.join(input_folder, fmt)))

    for path in img_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, gray)
        print(f"保存灰度图：{save_path}")

    print("批量灰度转换完成！")


# ===== 示例调用 =====
if __name__ == "__main__":
    batch_convert_to_gray("input_images", "output_gray")

```