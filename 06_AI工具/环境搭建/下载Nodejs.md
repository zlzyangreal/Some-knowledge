# 1. 下载
[下载链接](https://nodejs.org/en/download)
下载安装 Windows 安装程序(.msi)
# 2.环境配置
右键此电脑-----属性-----系统-----系统信息-----高级系统设置 -----高级------环境变量
在系统变量里**新建一个NODE_HOME**，变量值为安装路径（例如我的:**D：\software\node**）
然后再在系统变量的【**path**】中添加

**%NODE_HOME%**
**%NODE_HOME%\node_global**
**%NODE_HOME%\node_cache**

然后将用户变量path默认的C:\User\    \AppDate\Roaming\npm 改成安装路径（**D:\software\node\node global**）