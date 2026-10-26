# 1.在工程下写shell文件
```shell
#!/bin/bash
SELF_PATH=$(realpath -s "$0")

if [[ "$SUDO_ASKPASS" = "$SELF_PATH" ]]; then
    zenity --password --title="$1"
else
    exec env SUDO_ASKPASS="$SELF_PATH" sudo -A /usr/bin/gdb $@
fi
```
# 2.修改launch.json
```json
 "miDebuggerPath": "${workspaceFolder}/gdb_root.sh"
```
# 3.加权
```bash
sudo chmod +x gdb_root.sh
```