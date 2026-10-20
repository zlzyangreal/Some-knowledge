# 1. 总结
![[图像复原与重建1.png]]
# 2. Code
```python
# 运动模糊核
def motion_kernel(L, theta_deg):
    """L: 模糊长度 (pixels), theta_deg: 方向角"""
    size = max(L, 15)
    kernel = np.zeros((size, size))
    center = size // 2
    theta = np.deg2rad(theta_deg)
    for i in range(L):
        x = i
        y = int(round(i * np.tan(theta)))
        if abs(y) <= center:
            kernel[center + y, center + x] = 1
    kernel /= L
    return kernel

# 散焦核
def defocus_kernel(R):
    """R: 模糊半径"""
    size = 2*R + 1
    kernel = np.zeros((size, size))
    y, x = np.ogrid[-R:R+1, -R:R+1]
    mask = x*x + y*y <= R*R
    kernel[mask] = 1
    return kernel / kernel.sum()

# 高斯核
def gaussian_kernel(ksize, sigma):
    return cv2.getGaussianKernel(ksize, sigma) @ cv2.getGaussianKernel(ksize, sigma).T
```
