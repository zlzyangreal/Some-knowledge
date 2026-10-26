# VScode全代码调试

> “全代码调试”= 能在一整个工程里打断点、单步、看调用栈，而不是只跑单个文件。
> 关键就两份文件：`.vscode/tasks.json`（怎么编译）和 `.vscode/launch.json`（怎么启动调试）。

## 1. 前置条件

- 装了 C/C++ 扩展（ms-vscode.cpptools）。
- 编译器在 PATH 里：Windows 用 MinGW-w64 或 MSVC，Linux 用 gcc/g++。
- 工程能编过：先保证 `g++ main.cpp -o main -g` 在终端能跑通，再配调试。

## 2. tasks.json：先定“怎么编译”

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "build",
      "type": "shell",
      "command": "g++",
      "args": [
        "-g", "-O0", "-std=c++17",
        "${workspaceFolder}/src/main.cpp",
        "${workspaceFolder}/src/*.cpp",
        "-I${workspaceFolder}/include",
        "-o", "${workspaceFolder}/build/main.exe"
      ],
      "group": { "kind": "build", "isDefault": true },
      "problemMatcher": ["$gcc"]
    }
  ]
}
```

- `-g` 生成调试符号，**没有它断点全是空心圆**（灰色不可用）。
- `-O0` 关优化，否则变量会被优化掉、单步乱跳。
- `problemMatcher: $gcc` 让编译错误显示在“问题”面板并可点击跳转。

## 3. launch.json：再定“怎么启动”

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "调试 main",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/main.exe",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "miDebuggerPath": "C:\\mingw64\\bin\\gdb.exe",
      "preLaunchTask": "build",
      "setupCommands": [
        { "description": "为 gdb 启用整齐打印", "text": "-enable-pretty-printing", "ignoreFailures": true }
      ]
    }
  ]
}
```

关键字段：

| 字段 | 说明 |
| --- | --- |
| `program` | 要调试的**可执行文件**（不是源码） |
| `preLaunchTask` | 启动前先跑的任务名，必须和 tasks.json 的 `label` 一致 |
| `miDebuggerPath` | gdb 路径，Linux 填 `/usr/bin/gdb` |
| `args` | 传给程序的命令行参数 |
| `cwd` | 工作目录，影响程序里的相对路径 |
| `externalConsole` | 需要 `cin` 交互时设 `true` |

## 4. 调试操作

| 操作 | 快捷键 | 说明 |
| --- | --- | --- |
| 断点 | `F9` | 红点=有效，灰圈=没加载符号 |
| 启动 | `F5` | 走 launch.json |
| 单步跳过 | `F10` | 不进入函数 |
| 单步进入 | `F11` | 进入函数 |
| 跳出 | `Shift+F11` | 返回上一层 |
| 停止 | `Shift+F5` | |

侧边栏能看到：局部变量、监视表达式、调用栈、断点列表。调试控制台可执行 `-exec print var` 之类 gdb 命令。

## 5. 多文件 / 多目标工程

- 每个可执行目标配一份 configuration，用 `name` 区分。
- 用 CMake 时不要手写编译命令，改用 CMake Tools 扩展：
  `CMake: Configure` → `CMake: Set Debug` → 直接在状态栏点虫子图标，它会自动生成调试配置。
- 库代码进不去：把 `justMyCode`/`skipFiles` 配好，或确认库带调试符号。

## 6. 常见问题

| 现象 | 原因 |
| --- | --- |
| 断点是灰圈、点了不停 | 编译没加 `-g`，或 program 指向的不是刚编出来的文件 |
| `Unable to start debugging` | `miDebuggerPath` 路径错，或 gdb 不在 PATH |
| 程序一闪而过 | `externalConsole: false` 时终端在 VS Code 内，程序结束后自动关；加断点或 `system("pause")` |
| 变量显示 `<optimized out>` | 开了 `-O2`，改回 `-O0` |
| 中文乱码 | Windows 控制台编码问题，`chcp 65001` 或源码存 UTF-8 |

## 相关

- [[launch.json文件]]、[[tasks.json文件]]、[[c_cpp_properties.json文件]]
- [[setting.json文件]]、[[Windows上VScode使用VS的编译器]]
- [[python调试]]、[[gdb调试器]]
- [[远程连接]]
