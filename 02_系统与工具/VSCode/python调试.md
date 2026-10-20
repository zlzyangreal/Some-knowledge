# python调试

> VS Code 调试 Python，本质是配一份 `launch.json`，让调试器以正确的方式（参数、环境、工作目录）启动进程。

## 1. 最省事：直接按 F5

第一次按 F5 时 VS Code 会问“选择调试配置”，选 **Python File**，它会自动生成：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: 当前文件",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

`type` 在新版是 `debugpy`，老版本是 `python`。

## 2. 常用配置项

```json
{
  "name": "训练脚本",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/train.py",
  "args": ["--epochs", "10", "--data", "data/coco"],
  "cwd": "${workspaceFolder}",
  "env": { "CUDA_VISIBLE_DEVICES": "0" },
  "envFile": "${workspaceFolder}/.env",
  "console": "integratedTerminal",
  "justMyCode": true
}
```

- `program`：入口文件，`${file}` 表示当前打开的文件
- `args`：命令行参数，等价于终端里敲在后面的东西（注意每个参数分开写）
- `cwd`：工作目录，**相对路径读文件出错基本都是这里没设对**
- `justMyCode: true`：只在自己代码里停，不跳进库源码

## 3. 断点与调试操作

| 操作 | 快捷键 |
| --- | --- |
| 切换断点 | `F9` |
| 开始/继续 | `F5` |
| 单步跳过（不进函数） | `F10` |
| 单步进入 | `F11` |
| 单步跳出 | `Shift+F11` |
| 重启调试 | `Ctrl+Shift+F5` |

常用技巧：

- **条件断点**：右键断点 → 编辑条件，如 `i == 100`，避免循环里点几十次。
- **日志断点**：右键 → 编辑日志消息，填 `i = {i}`，不中断只打印。
- **调试控制台**：可以临时执行表达式，比如 `len(data)`、`model.state_dict().keys()`。
- **监视窗口**：把 `loss`、`batch["image"].shape` 加进去，每步都能看。

## 4. 远程 / 容器里调试

```json
{
  "name": "远程 Python",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "192.168.1.10", "port": 5678 },
  "pathMappings": [
    { "localRoot": "${workspaceFolder}", "remoteRoot": "/home/user/project" }
  ]
}
```

远端要先装 `debugpy` 并以监听模式启动：

```bash
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client train.py
```

## 5. 不用 VS Code 时的备选

```python
breakpoint()            # Python 3.7+，等价 import pdb; pdb.set_trace()
# pdb 常用：n 下一步、s 进入、c 继续、p 变量、q 退出
```

## 相关

- [[launch.json文件]] — 字段总览（C++/通用）
- [[tasks.json文件]] — 调试前先跑构建任务
- [[VScode全代码调试]]
- [[python]]、[[常用库索引]]
