# 1.简介
Skills 本质上就是教 AI 按固定流程做事的操作说明书，一旦写好，就能像函数一样反复调用。Skills 以 Markdown 文件形式书写
![[skills.svg]]
Skills 和传统 Prompt 最大的区别是：按需加载 + 渐进式披露（只在需要时才把厚厚的 SOP 塞进上下文，极大节省 token）。
# 2. Skill 的核心结构
Skills 的核心就是：一个文件夹 + 一个 SKILL.md 文件
**SKILL.md 文件包含：**
- 元数据（至少要有名称和描述）
- 告诉 AI 如何完成某一特定任务的指令
一个 Skill 本质上就是一个 Markdown 文件（**文件名固定为 SKILL.md**）
`SKILL.md` 是由 YAML 头部和 Markdown 指令部分组成。
### 2.1 C++程序员模板
```makedown
---
name: cpp-code-assistant
description: 编写 C++ 核心逻辑，配置跨平台构建系统，并进行代码调试与诊断
---

# C++ 代码助手

## 使用场景
当需要进行 C++ 程序开发、测试或错误排查时使用，例如：

- 实现高性能算法（如复杂数据结构处理或 LeetCode 风格题目）
- 编写并集成特定领域的处理模块（如计算机视觉中的 OpenCV 调用）
- 诊断 C++ 运行时的内存错误或定位段错误

## 代码构建与编译
- 针对单文件算法测试，使用 `g++` 或 `clang++` 进行快速编译执行  
- 针对多文件工程，自动生成、配置和修改 `CMakeLists.txt`  
- 适配不同开发环境的编译工具链（如 Linux 系统环境或 Windows 下的 MinGW-w64）  

## 调试与诊断
- 配合 `GDB` 分析堆栈信息，定位程序崩溃的具体代码行  
- 启用编译选项 `AddressSanitizer (-fsanitize=address)` 检查指针越界和内存泄漏  
- 结合 Git 追踪代码变更历史，辅助排查引入 Bug 的具体提交

```
## 2.2 研究深度学习的学者，投递论文模板
```makedown
---
name: dl-paper-submission
description: 辅助深度学习学术论文的实验数据整理、LaTeX 排版校对、文献引用管理及投稿前检查
---

# 论文投稿助手

## 使用场景
当需要准备、撰写或修改深度学习学术论文并进行会议/期刊投稿时使用，例如：

- 整理模型训练日志并生成消融实验对比表格
- 检查论文的格式规范与参考文献列表
- 润色学术英语表达（如摘要与方法论描述）

## 实验数据处理
- 解析框架运行产生的日志或 JSON 文件，自动提取 mAP、FPS 等关键评价指标  
- 自动化生成对比不同算法模块（如各类注意力机制或跟踪算法）性能的 LaTeX 表格代码  

## 排版与校对
- 基于目标期刊或会议（如 IEEE 格式）检查双栏排版、图表编号及复杂的数学公式环境  
- 验证 `BibTeX` 参考文献列表的完整性，排查文中引用与文末列表的对应错误或格式遗漏
```

## 2.3 总结
claude 给的开头
```makedown
---
name: 
description: 
license: Apache-2.0
metadata:
  author: example-org
  version: "1.0"
---
```

| 字段            | 必需  | 说明                                                |
| ------------- | --- | ------------------------------------------------- |
| name          | 是   | Skill 名称，最长 64 字符，只能使用小写字母、数字和 `-`，且不能以 `-` 开头或结尾 |
| description   | 是   | 功能与使用场景说明，最长 1024 字符，不能为空                         |
| license       | 否   | 许可证名称或指向随 Skill 附带的许可证文件                          |
| compatibility | 否   | 环境与依赖说明（产品、系统包、网络权限等），最长 500 字符                   |
| metadata      | 否   | 自定义键值对，用于扩展元数据（如作者、版本号）                           |
| allowed-tools | 否   | 允许使用的工具列表（空格分隔，实验性功能）                             |
## 2.4 `SKILL.md` 正文的“Prompt 编写准则”

- **设定边界 (Constraints)：** 明确告诉 AI “不要做什么”。例如在 C++ 开发中，明确规定“必须保证没有内存泄漏，优先推荐智能指针而非裸指针”。
- **规定输入输出格式 (I/O Formatting)：** 就像你笔记里的注释专家例子一样，严格要求 AI 先输出完整代码，再输出 diff。
- **异常处理策略 (Fallback)：** 告诉 AI 当遇到找不到的变量或不确定的逻辑时该怎么做（例如：“如果日志解析失败，请停止执行并请求用户提供完整的 `.log` 文件样本”）。
# 3. 项目结构
```txt
my-skill/
├── SKILL.md      # 必需：指令 + 元数据
├── scripts/      # 可选：可执行代码
├── references/   # 可选：文档资料
└── assets/       # 可选：模板、资源
```
![[skills-runoob-123.jpg]]

## 3.1 AI 工作过程了解
技能（Skills）的核心优势在于利用**渐进式加载**（Progressive Disclosure）来极其高效地管理 AI 的上下文窗口（Context Window），从而在不耗尽 Token 的前提下处理复杂的工程项目：
- **第一阶段：轻量发现 (Discovery)**：启动时，AI 只把所有技能的 `name`、`description` 和（如果有的话）触发词加载到系统后台。此时每个技能只占极少量 Token，相当于 AI 只扫视了一遍“工具箱的标签”。
- **第二阶段：精准激活 (Activation)**：当你的 Prompt（例如“帮我写个 CMakeLists 并排查一下段错误”）与某个技能的描述高度匹配时，AI 才会将该技能对应的完整 `SKILL.md` 指令文本动态“塞入”当前的活跃上下文中。
- **第三阶段：深度挂载 (Context Mounting)**：AI 读取 `SKILL.md` 后，如果指令中要求参考外部规范（如 `references/` 下的算法公式）或使用特定模板（如 `templates/` 下的代码骨架），AI 会根据当前任务按需读取这些附属文件。
- **第四阶段：受控执行 (Execution)**：AI 严格按照 Skill 设定的边界、流程（SOP）和输出格式执行任务。如果配置了权限，AI 甚至会主动运行 `scripts/` 下的脚本（比如跑个 Python 脚本处理日志）来辅助作答。
- **第五阶段：状态卸载 (Unloading)**：当你的对话意图转移到完全无关的话题时，该技能的详细指令会被移出上下文，释放空间。
## 3.2 简单 AI 工作流程
Claude Code 按以下顺序查找并加载 Skill（越具体的位置优先级越高）：

|级别|路径|生效范围|
|---|---|---|
|企业级|通过管理控制台配置（managed settings）|组织内所有用户|
|个人级|`~/.claude/skills/<skill-name>/SKILL.md`|你所有项目|
|项目级|`.claude/skills/<skill-name>/SKILL.md`|仅当前项目|
|插件级|`<plugin>/skills/<skill-name>/SKILL.md`|启用该插件的环境|
每个 Skill 就是一个文件夹，文件夹名即技能标识（推荐 kebab-case 小写+连字符）。
### 3.2.1 C++ 代码助手 (cpp-code-assistant) 最简结构

最简结构：
```makedown
~/.claude/skills/
  └── cpp-code-assistant/           # 技能文件夹名
      └── SKILL.md                  # 唯一必填文件（必须全大写 + .md 小写）
```
SKILL.md 完整格式：
```makedown
---
name: cpp-code-assistant
description: >-
  编写 C++ 核心逻辑，配置跨平台构建系统，并进行代码调试与诊断。
  适合处理高性能算法、配置工程以及内存泄漏排查。
trigger_keywords:
  - c++
  - cmake
  - gdb
  - 内存泄漏
  - 段错误
  - 编译报错
---

# C++ 核心开发助手

## 核心原则
- 优先推荐现代 C++ 容器与特性，避免不安全的裸指针操作。
- 提供编译和环境配置指令时，需同时兼顾 Linux 环境和 Windows (`MinGW-w64`) 环境的差异。
- 遇到算法逻辑问题时，优先考虑时间与空间复杂度。

## 执行流程（严格遵守）
1. **诊断阶段**：如果是报错，先指导使用 GDB 分析堆栈，或提示开启 `-fsanitize=address` 编译选项。
2. **代码生成**：输出包含清晰注释的完整 C++ 代码，并在关键算法部分（如谷歌风格逻辑或 OpenCV 接口调用）解释设计思路。
3. **构建配置**：如果涉及多文件工程，必须自动提供配套的 `CMakeLists.txt`。

现在直接开始处理用户提供的 C++ 问题，不要闲聊。
```
#### 进阶文件结构
当 C++ 工程变得复杂（例如开发包含多个模块的目标跟踪系统），需要引入设计规范和诊断脚本时，推荐以下组织方式：
```makedown
~/.claude/skills/cpp-code-assistant/
  ├── SKILL.md                  # 核心指令 + 元数据
  │
  ├── templates/                # 工业级代码模板
  │   ├── gtest-fixture.cpp.md  # Google Test 的标准测试夹具 (Fixture) 模板
  │   └── abseil-status.cpp.md  # 使用 absl::Status/absl::StatusOr 进行现代错误处理的模板
  │
  ├── references/               # 权威规范与规则
  │   ├── google-cpp-style.md   # Google C++ 编程规范核心原则 (头文件包含顺序、命名约定等)
  │   └── linux-posix-api.md    # Linux 核心系统调用 (epoll, 进程间通信) 与 POSIX 线程安全最佳实践
  │
  └── scripts/                  # 自动化检查脚本
      ├── run-clang-tidy.sh     # 静态代码分析与现代 C++ 特性 (Modernize) 检查
      └── run-clang-format.sh   # 依据 .clang-format 文件自动格式化代码风格
```
在 SKILL.md 中引用方式示例:
```makedown
# C++ 核心开发助手

## 核心原则
- 严格遵循 Google C++ Style Guide，禁止使用 C 风格的类型转换，指针所有权必须通过 `std::unique_ptr` 等明确交接。
- 处理异常和返回值时，优先推荐使用类似 Google Abseil 库的 `absl::Status` 或 `std::expected` (C++23) 进行错误传播。
- 涉及底层系统编程时，确保对 Linux POSIX 接口的正确调用（例如处理 `EINTR` 中断信号，以及正确管理文件描述符生命周期）。

## 附属文件调用指引（Claude 执行时参考）
- 编写单元测试时，请严格参考 `templates/gtest-fixture.cpp.md` 设置 `TEST_F` 与 `SetUp/TearDown` 内存隔离。
- 遇到系统级 I/O 或并发处理疑问时，请对照 `references/linux-posix-api.md` 确保多线程操作的原子性与无锁设计规范。
- 每次输出大型代码块后，主动提示用户可运行 `scripts/run-clang-tidy.sh "{源文件}"` 扫描潜在的内存越界或悬垂引用。
```
### 3.2.2 深度学习论文助手 (dl-paper-submission)
**最简结构** 适合快速排查单篇论文的格式或润色摘要。
```txt
~/.claude/skills/
  └── dl-paper-submission/          # 技能文件夹名
      └── SKILL.md                  # 唯一必填文件
```
SKILL.md 完整格式
```makedown
---
name: dl-paper-submission
description: >-
  辅助深度学习学术论文的实验数据整理、LaTeX 排版校对、文献引用管理及投稿前检查。
trigger_keywords:
  - 论文润色
  - latex
  - 排版
  - 实验数据
  - bibtex
---

# 论文投稿助手

## 核心原则
- 保持客观、严谨的学术英语口吻，避免口语化表达。
- 重点关注目标检测与跟踪领域的专业术语准确性（如几何补偿、各种注意力机制）。

## 执行流程（严格遵守）
1. **实验数据处理**：根据用户提供的实验日志，自动提取关键评价指标，并生成对比不同算法模块（如 YOLO26 结合 BiFormer、GAM 或 DSConv）性能的 LaTeX 表格代码。
2. **排版与校对**：基于 IEEE 等标准格式，检查双栏排版、图表编号及复杂的数学公式环境是否正确闭合。
3. **文献管理**：验证 `BibTeX` 参考文献列表的完整性，排查文中引用与文末列表的对应错误或格式遗漏。
```
#### **进阶文件结构** 
当需要整理包含大量消融实验结果（如各种注意力模块对比）、撰写格式严格的毕业论文或期刊论文时，推荐以下组织方式：
```txt
~/.claude/skills/dl-paper-submission/
  ├── SKILL.md                  # 核心指令 + 元数据
  │
  ├── templates/                # 常用模板
  │   ├── ieee-table.tex.md     # IEEE 期刊双栏排版的标准 LaTeX 表格模板
  │   └── gc-deepsort-method.md # GC-DeepSORT 系统方法论章节的结构骨架(这里是自己论文的章节安排)
  │
  ├── references/               # 规范、规则
  │   ├── cv-metrics.md         # 目标检测与跟踪指标的严格计算定义
  │   └── journal-rules.md      # 目标期刊的特殊格式要求与 BibTeX 规范
  │
  └── scripts/                  # 可执行脚本
      ├── extract-metrics.py    # 从训练日志中自动提取评价指标的 Python 脚本
      └── format-bibtex.py      # 清洗和标准化 BibTeX 条目的脚本
```
在 SKILL.md 中引用方式示例
```makedown
当需要生成不同注意力机制的消融实验对比表格时，请严格参考 `templates/ieee-table.tex.md` 的 LaTeX 语法规范。

如果用户要求分析跟踪系统的性能评价指标，请对照 `references/cv-metrics.md` 确认计算定义是否准确。

如需解析长篇训练日志提取数据，可主动执行脚本 `scripts/extract-metrics.py "{日志文件路径}"` 进行数据清洗。
```
