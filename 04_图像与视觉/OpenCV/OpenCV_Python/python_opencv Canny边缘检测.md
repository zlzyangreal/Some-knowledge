# 1.Code
```python
import cv2
import os

def canny_edge_detect(input_path, output_dir, low_threshold=50, high_threshold=150):
    """
    使用Canny算法进行边缘检测，并将结果保存到指定路径。
    
    参数:
        input_path (str): 输入图像路径
        output_dir (str): 输出结果文件夹
        low_threshold (int): Canny低阈值
        high_threshold (int): Canny高阈值
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像并转为灰度
    img = cv2.imread(input_path)
    if img is None:
        print(f"[ERROR] 无法读取图像: {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯平滑去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

    # Canny边缘检测
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    # 生成保存路径
    output_path = os.path.join(output_dir, "edges_canny.png")

    # 保存结果
    cv2.imwrite(output_path, edges)
    print(f"[INFO] Canny 边缘检测结果已保存到: {output_path}")

if __name__ == "__main__":
    input_image = "test_input.jpg"
    output_folder = "./canny_output"
    canny_edge_detect(input_image, output_folder, low_threshold=60, high_threshold=180)

```
![[python_opencv_Canny滤波.png]]