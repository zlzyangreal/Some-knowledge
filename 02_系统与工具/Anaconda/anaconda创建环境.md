# 创建环境
## 操作步骤
* 1.打开命令提示符或Anaconda Prompt。
* 2.运行以下命令来创建一个新的环境并指定安装路径：
```bash
conda create --prefix D:\ProgramData\anaconda3\envs\YOLO_LK python=x.x
```
* 3.当提示是否继续安装时，输入`y`并按回车键确认。
* 4.环境创建完成后，使用以下命令激活该环境：
```
conda activate D:\ProgramData\anaconda3\envs\YOLO_LK
```
## 相关指令

### 默认路径建立环境(**该方法在Linux上用的比较多**)：
```bash
conda create -n your_env_name python=x.x
```
### 退出当前环境：
```bash
conda deactivate
```
### 激活环境
#### 默认路径
```bash
conda activate yolov7test
```
#### 指定路径
```bash
conda activate D:\xx\xx
```
### 查看所有已建环境
```bash
conda info --envs
```

### 删除环境
```bash
conda remove -n xxxxx(名字) --all
```

# 环境克隆

```bash
conda create -n B  --clone A
```
将A克隆到B
# 导出环境
```bash
conda env export > environment.yml
```