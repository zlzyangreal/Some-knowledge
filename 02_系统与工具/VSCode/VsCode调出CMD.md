![[Snipaste_2025-09-21_14-23-39.png]]

## 三种“调出命令行”的方式

### 1. 集成终端（最常用）

| 快捷键 | 作用 |
| --- | --- |
| `` Ctrl+` `` | 打开/关闭终端 |
| `Ctrl+Shift+`` ` `` | 新建终端 |
| `` Ctrl+Shift+5 `` | 拆分终端 |

默认 shell 在 Windows 上是 PowerShell。想默认开 cmd：

```json
// settings.json
{
  "terminal.integrated.defaultProfile.windows": "Command Prompt",
  "terminal.integrated.profiles.windows": {
    "Command Prompt": { "path": "C:\\Windows\\System32\\cmd.exe" }
  }
}
```

### 2. 在文件所在目录打开外部 CMD

资源管理器里对文件夹 `Shift + 右键` → “在此处打开 PowerShell/命令提示符”；
或 VS Code 里 `右键文件 → 在文件资源管理器中显示`，再在那个窗口的地址栏输入 `cmd` 回车。

### 3. 用任务直接跑命令

`.vscode/tasks.json`：

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "打开 cmd",
      "type": "shell",
      "command": "start cmd",
      "problemMatcher": []
    }
  ]
}
```

`Ctrl+Shift+P` → `Tasks: Run Task` → 选“打开 cmd”，就会弹出独立窗口。

## 上图说明

上图是 `Ctrl+Shift+P` 命令面板里搜索 terminal 的结果，可以看到 “Terminal: Create New Terminal”
和 “Terminal: Run Selected Text in Active Terminal” 两项——后者可以把编辑器里选中的命令直接丢进终端执行，调试时很好用。

## 相关

- [[vscode常用快捷键]]
- [[tasks.json文件]]
- [[setting.json文件]]
