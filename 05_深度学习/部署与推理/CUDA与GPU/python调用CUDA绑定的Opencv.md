```python
import sys
import os

# --------------------------------------------------------------------------------
# 🔑 关键步骤：手动添加编译好的 OpenCV 路径
# 路径指向 build/lib/python3 (这里面存放着 cv2.cpython-38-x86_64-linux-gnu.so)
# --------------------------------------------------------------------------------
cv2_build_path = '/home/alphafox/Desktop/SDK/opencv/opencv-4.10.0/build/lib/python3'

if os.path.exists(cv2_build_path):
    # 插入到系统路径的最前面，确保优先加载如果你系统里还有其他版本的cv2
    sys.path.insert(0, cv2_build_path) 
    print(f"✅ 已手动加载 OpenCV 路径: {cv2_build_path}")
else:
    print(f"❌ 警告: 路径不存在，可能会报错 -> {cv2_build_path}")
# --------------------------------------------------------------------------------

import cv2
```
