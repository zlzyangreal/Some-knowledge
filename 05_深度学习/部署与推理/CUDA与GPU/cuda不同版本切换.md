在 /usr/local 目录下软连接cuda
- 如果下述情况连着`alternatives`链接一起删除更改
```bash
ls -l /usr/local/cuda
```
```bash
  (base) alphafox@alphafox:/usr/local$ ls -l /usr/local/cuda
lrwxrwxrwx 1 root root 22 10月 22 22:52 /usr/local/cuda -> /etc/alternatives/cuda
  ```
1. 删除原来的cuda软连接
```bash
sudo rm -rf /etc/alternatives/cuda # 上诉情况加上
sudo rm -rf /usr/local/cuda
```
2. 连接新的cuda版本
```bash
sudo ln -s /usr/local/cuda-11.6 /etc/alternatives/cuda # 上诉情况加上
sudo ln -s /usr/local/cuda-12.3 /usr/local/cuda
```
* 以12.3版本为例
3. 查看当前cuda版本
```bash
nvcc --version
```
4. 改动系统变量
在`./bashrc`文件
```bash
# CUDA environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
然后刷新环境：
```bash
source ~/.bashrc
```