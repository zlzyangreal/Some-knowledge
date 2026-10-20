```python
import sys
import os
import ctypes

# --------------------------------------------------------------------------------
# 🚑 紧急修复：GLIBCXX 版本缺失问题
# 强制预加载 Anaconda 环境中的 libstdc++.so.6
# --------------------------------------------------------------------------------
# 这是您报错信息中显示的 Anaconda 环境路径
conda_env_lib = '/home/alphafox/Desktop/APP/anaconda3/envs/vascular2/lib'
libstdc_path = os.path.join(conda_env_lib, 'libstdc++.so.6')

if os.path.exists(libstdc_path):
    try:
        # mode=ctypes.RTLD_GLOBAL 确保后续导入的库（如 llvmlite）能看到这个版本
        ctypes.CDLL(libstdc_path, mode=ctypes.RTLD_GLOBAL)
        print(f"✅ 成功强制预加载 Conda 的 libstdc++: {libstdc_path}")
    except OSError as e:
        print(f"❌ 预加载失败，依然可能会报错: {e}")
else:
    print(f"❌ 警告: 在 {conda_env_lib} 下没找到 libstdc++.so.6，请确认路径是否正确")

from numba import cuda
```