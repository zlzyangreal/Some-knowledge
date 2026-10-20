# 板卡(服务器)远程设置
## 1.设置ssh的sshd_config文件
```bash
cd /etc/ssh
sudo vim sshd_config
```
![[Screenshot 2024-06-14 205712.png]]
## 2.查看
```bash
echo $DISPLAY
```
# 本地设置
在vscode加入指的是`launch.json`
```json
           // 环境变量
           "environment": [
            {
               "name": "DISPLAY", "value": "localhost:10.0"
            }
           ],
```
## 对于python加入
```python
import os
os.environ["DISPLAY"] = "localhost:10.0"
```