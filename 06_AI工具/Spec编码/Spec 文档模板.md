### 1. 标准操作流程 (SOP)
不要一上来就让 AI 写代码，按这 4 步走：
1. **Draft (起草)**：
    - 新建一个 `spec.md` 文件。
    - 用中文+伪代码，把你要做的功能、数据结构写清楚（就像写需求文档）。
2. **Context (投喂)**：
    - 把这个 `spec.md` 扔给 AI（Cursor/Claude/ChatGPT）。
3. **Prompt (指令)**：
    - 输入：_"Read **@spec.md** strictly. Implement the code following the design."_
    - (中文：严格阅读文档，完全按设计实现，不要自由发挥。)
4. **Review (验收)**：
    - 如果代码不对，**不要直接改代码**，去改 `spec.md`，然后让 AI 重新生成。
### 2. 万能 Spec 文档模板 (复制即用)
这是 Spec Coding 的灵魂。每次写代码前，把下面这个填空填好，甩给 AI 即可。
Markdown
```
# Spec: [项目/功能名称]

## 1. Goal (目标)
简述这个功能要做什么。（例如：读取视频文件夹，用 YOLO 检测并输出日志）

## 2. Tech Stack (技术硬性要求)
* **语言**: Python 3.9 / C++ 17
* **关键库**: Ultralytics, OpenCV, NumPy
* **约束**: (例如：禁止使用 Class，只用函数；或者必须写 Type Hints)

## 3. Data Structure (数据结构 - 最重要！)
这里定义死，AI 就不会乱写。
* **Input**: (例如：MP4文件路径 list)
* **Output**: (例如：CSV文件，包含 timestamp, class_id)
* **Core Objects**:
    ```python
    @dataclass
    class DetectionResult:
        frame_id: int
        box: list[float]
        conf: float
    ```

## 4. Workflow (逻辑步骤)
1.  初始化模型。
2.  遍历文件。
3.  异常处理（如果文件坏了怎么办）。
4.  保存结果。

## 5. Implementation Rules (代码规范)
* 必须包含异常捕获 (try-except)。
* 必须添加中文注释。
```
## 3.实例
```makedown
# Spec: Simple Camera Viewer

## 1. Goal
创建一个 Python 脚本，调用电脑摄像头，实时显示画面，直到用户按下退出键。

## 2. Tech Stack
* **Language**: Python 3.9+
* **Library**: opencv-python (cv2)

## 3. Implementation Details
* **Camera Index**: Default to `0`.
* **Window Name**: "My Camera Feed"
* **Quit Key**: Press 'q' to close the window and release resources.

## 4. Workflow
1. Initialize VideoCapture.
2. Check if camera is opened successfully; if not, print error and exit.
3. Start a loop to read frames.
4. Display frame in a window.
5. Wait for 'q' key press to break loop.
6. Release resources and destroy windows.
```
## 4. 改bug
AI 的失误（Spec 没错，是 AI 写崩了）

**症状**：Spec 文档写得很完美，逻辑通顺，但 AI 生成的代码报了 `SyntaxError`，或者漏了 import，或者变量名写错了。 **处理原则**：**把报错信息当成反馈（Feedback），扔回给 AI。**

**操作步骤（以 OpenCV 为例）：**

1. **运行报错**： 假设运行 `main.py` 时终端报错：
    

    ```bash
    cv2.error: OpenCV(4.9.0) .../imgproc.cpp: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'
    ```
    
    _(原因：摄像头没打开，frame 是空的，但代码还尝试转灰度)_
    
2. **投喂报错（不改 Spec，只报 Bug）**： 把**Spec文档** + **当前代码** + **错误信息** 一起给 AI（如果你用 Cursor/Cline，它自动能看到代码，你只需给错误信息）。
    
    **Prompt（指令）：**
    
    > "I got this error when running the code: `[粘贴报错信息]` The Spec is correct. Please fix the code to handle this edge case (frame is None) strictly following the existing Spec." _(代码报错了。Spec 是对的。请修复代码以处理这个边界情况，严格遵循现有 Spec。)_
    
2. **结果**：AI 会根据错误信息自我修正代码（比如加个 `if frame is None: break`）。

## 5. 加新内容
不要把所有东西写在一个文件里。
- `base_spec.md`: 核心架构、技术栈、通用规则。
- `feature_screenshot_spec.md`: 专门描述截图功能的细节。

投喂时：_"Based on @base_spec.md, implement the feature defined in @feature_screenshot_spec.md"_