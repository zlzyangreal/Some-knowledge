# 1. Windows

## 安装
1. 打开 Anaconda 终端
    * 验证是否安装成功，以及版本指令`pip --version`
2. 安装依赖的第三方库
    * `pip install PyQt5`
    * `pip install PyQt5-tools`
    * `pip install lxml`
3. 安装插件 `pip install labelimg`

打开路径

![[Labelimg路径.png]]
* 本机路径 C:\Users\86132\AppData\Roaming\Python\Python311\Scripts

## 使用
labelimg的标注模式分为VOC和YOLO两种，两种模式下生成的标注文件分别为.xml文件和.txt文件，因此在进行标注前需要优先选择好标注的模式
### 简单解释
1. 打开软件后先打开数据集目录
2. 在 labels 目录创建 classes.txt，内容是哪些种类
3. 选择模式Yolo，选择标签文件夹

操作界面

![[Labelimg操作界面.png]]
* labelimg左侧菜单栏中按钮功能

   ![[labelimg左侧菜单栏中按钮功能.png]]
* 在labelimg中，标注的快捷键为w，标注后用鼠标拖动锚框进行框选，框选完毕后会弹出类别选择框，如果是当前已有类则直接选择即可，若需要新加类别则在输入框中输入类别标签并确定

   ![[Labelimg标签操作.png]]

## 两种文件和类别文件
xml文件
* 在 [[VOC资料]] 有

txt文件(YOLO备注模式)
* 数分别表示类别，中两个中心点坐标，后两个w，h

类别文件
* classes文件中是标注的类别
# 2. Ubuntu 源码
- 与普通py包一样，在虚拟环境中运行
## 2.1 [下载源码](https://github.com/HumanSignal/labelImg)
## 2.2 下载Labelimg必要依赖
```bash
cd requirements/
pip3 install -r requirements-linux-python3.txt
```
## 2.3 下载运行必要驱动
```bash
# 如果有pyQT残余文件，先把 pip 装的 PyQt5 彻底删干净（很重要！）
pip uninstall pyqt5 pyqt5-qt5 pyqt5-sip qt-pyqt5 -y
# 用 conda 重新安装（会自动匹配正确的 Qt 运行时库） 
conda install pyqt=5.15.7 -c conda-forge -y
# 再装 lxml（labelImg 依赖） 
conda install lxml -c conda-forge -y
```
## 2.4 运行
在Git代码下运行
```bash
python labelimg.py
```