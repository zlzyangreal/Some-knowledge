# 1. Agent skills是什么
Agent Skills 本质上是一个模块化的 Markdown 文件，能教会 AI 工具 （如 Claude、GitHub Copilot、Cursor 等） 执行特定任务，且支持自动触发、团队共享与工程化管理，彻底告别重复的提示词输入。
- [[Skills基础知识]]
- [[Gemini CLI下载和使用]]
## 1.1 **核心形式：**

- 一个 Skill 就是一个文件夹，里面必须有一个 SKILL.md 文件（包含说明和元数据），可选其他资源文件（如脚本、示例、参考文档）。
- Skill 是一个 Markdown 文件（SKILL.md），用于教 Claude 在特定场景下按你的方式做事。
- 本质是其实就是相当于给 AI 代理发放一本专业手册，AI 不会每次都从零学习，而是根据任务自动调用手册中的知识。
- 简单来说，过去我们用提示词（prompt）教 AI 做事，现在用 Agent Skills 可以把提示词 + 资源打包成可复用、可共享的技能包，更高效、更可靠。
# 2. Agent Skills 的工作原理

Agent Skills 的关键是渐进式披露，分三层加载：

- **层级 1：技能发现** -- AI 先读取所有技能的元数据（name 和 description），判断任务是否相关，这些元数据始终在系统提示中。
- **层级 2：加载核心指令** -- 如果相关，AI 自动读取 SKILL.md 的正文内容，获取详细指导。
- **层级 3：加载资源文件** -- 只在需要时读取额外文件（如脚本、示例），或通过工具执行脚本。
# 3. 实例
## 3.1 用C++助手写一个 Hello World （使用Gemini CLI）
### 步骤1. 构建 Skill 文件系统与配置

首先，在系统的全局 Skill 目录（例如 `/home/alphafox/.gemini/skills/`）下建立完整的工具链结构。
1. 创建目录结构（）：
```txt
目录：
..../.gemini/skills
创建
  ├── SKILL.md                  # 核心指令 + 元数据
  │
  ├── templates/                # 工业级代码模板
  │
  ├── references/               # 权威规范与规则
  │   ├── google-cpp-style.md   # Google C++ 编程规范核心原则 (头文件包含顺序、命名约定等)
  │
  └── scripts/                  # 自动化检查脚本
      └── run-clang-format.sh   # 依据 .clang-format 文件自动格式化代码风格
```
2. 写入核心配置文件 `SKILL.md`：
```makedown
---
name: cpp-code-assistant
description: 编写 C++ 核心逻辑，配置跨平台构建系统，并执行 Google C++ 规范检查。
trigger_keywords: [c++, cmake, google style]
---
# C++ 核心开发助手
严格遵循 `references/google-cpp-style.md` 的规范。
生成代码后，必须提示用户使用 `scripts/run-clang-format.sh` 格式化代码。
提供完整的 `CMakeLists.txt` 构建支持。
```
3. 写入规范文件 `references/google-cpp-style.md`：
```makedown
1. 包含头文件应按特定顺序（相关头文件, C 库, C++ 库, 其他库）。
2. 禁止使用 `using namespace std;`。
3. 以类的方式书写，类里面只写声明
```
4. 写入自动化脚本 `scripts/run-clang-format.sh`：
```shell
#!/bin/bash
clang-format -i -style=Google "$1"
echo "[代码格式化完成] 已应用 Google C++ Style: $1"
```
### 步骤3：Gemini CLI 触发 Skill 并生成代码
在目标工程目录（也就是上述项目目录） 中，通过 CLI 执行标准化指令。该指令需遵循“任务目标-输出要求-约束条件-后续动作”的四段式结构。
**执行内容：** 在系统终端输入以下完整的 CLI 命令
```bash
gemini --skill cpp-code-assistant \
--prompt "
[任务目标]：生成一个 Hello World 初始工程。
[局部定制要求]：
1. main.cpp 源代码需包含标准 Doxygen 格式的头部注释。
2. 附带的 CMakeLists.txt 需明确指定 C++20 标准。
[后续动作]：仅输出后续代码格式化与编译的 bash 命令，勿输出额外解释。
"
```
## 3.2 用论文助手看一篇论文
#### 步骤1：构建 Skill 文件系统与配置

在系统的全局 Skill 目录下建立论文助手的工具链结构。为了让 AI 每次解析论文都严格遵循“先计划后执行”和“特定格式   输出”的要求，我们将这些铁律固化到附属文件中。

目录结构：
```txt
~/.gemini/skills/dl-paper-assistant/
  ├── SKILL.md                  # 核心指令 + 元数据
  │
  ├── templates/                # 输出格式模板
  │   └── explanation-table.md  # 强制要求最终答案使用的表格排版模板
  │
  ├── references/               # 权威规范与规则
  │   ├── expert-persona.md     # 设定资深专家角色语气（浅显易懂但非口语化）
  │   └── workflow-rules.md     # 固化“先制定计划 -> 询问确认 -> 逐步执行”的严格流程
  │
  └── scripts/                  # 自动化处理脚本
      └── extract-pdf-images.py # 自动从 PDF 中提取 Figure 用于辅助讲解
```
##### a. 写入核心配置文件 `SKILL.md`：
```makedown
---
name: dl-paper-assistant
description: 以资深专家身份，深入浅出地讲解深度学习理论与算法设计。
trigger_keywords: [论文解析, 算法理论, 注意力机制]
---
# 深度学习论文解析助手

## 核心法则
1. **强制加载规范**：必须严格遵循 `references/workflow-rules.md` 执行“制定计划-确认-分步执行”的工作流，绝对不可在所有步骤未执行完前给出最终答案。
2. **角色与语调**：读取 `references/expert-persona.md`，语言需浅显易懂，具备学术严谨性，不可过度口语化。
3. **排版约束**：数学公式必须使用标准的 LaTeX 格式（如 $F_c$ 或独立公式块）。最终的完整解析必须以 `templates/explanation-table.md` 定义的表格形式呈现。
4. **图文并茂**：解析算法时，必须紧扣用户提供的论文原图（如网络结构图）进行对照讲解。
```
##### b. 最终输出格式约束：`templates/explanation-table.md`
此文件强制规范 AI 在完成所有分析步骤后，必须使用的高度结构化的表格格式。
```makedown
# 深度学习理论与算法解析汇总表

**指令：** 在完成所有分步解析并得到用户最终确认后，必须严格按照以下 Markdown 表格结构输出最终答案。表格内容需提炼前序步骤的核心精华，保持学术严谨性，不可过于简略，也不可堆砌冗余文字。

| 算法模块 / 理论概念 | 核心设计逻辑与理论支撑 | 核心数学推导 (LaTeX 格式) | 关联结构图 / 示意图 |
| :--- | :--- | :--- | :--- |
| **背景与动机** | (例如：现有注意力机制丢失了三维空间信息，限制了跨维度交互...) | N/A | 图 X (对比图) |
| **整体架构定义** | (简述输入张量的维度变化，以及整体信息流转过程) | $\mathbf{F}_{2} = \mathbf{M}_{c}(\mathbf{F}_{1}) \otimes \mathbf{F}_{1}$ | 图 X (整体流程图) |
| **核心模块 A (如：通道交互)** | (剥离代码，纯粹从算法设计角度解释该模块如何运作) | (填写对应的 LaTeX 公式，如维度置换操作) | 图 X (子模块图) |
| **核心模块 B (如：空间交互)** | (解释其与模块 A 的协同作用及理论创新点) | (填写对应的特征融合公式) | 图 X (子模块图) |
| **理论总结** | (平衡地总结该机制为何有效，对特征表示能力的具体提升在哪里) | N/A | 综合参考以上全图 |

**要求：**
- 数学公式必须使用严谨的 LaTeX 语法（如 $\mathbb{R}^{C \times H \times W}$）。
- 每一行必须明确指向论文中对应的 Figure 编号，确保图文对照。
```
##### c. 专家角色与语调约束：`references/expert-persona.md`
此文件用于把控 AI 的“人设”与语言风格，防止其出现 AI 常见的过度热情或口语化问题。
```makedown
# 资深深度学习专家设定与语调指南

## 核心身份
你是一位在计算机视觉和深度学习领域拥有深厚理论功底的资深研究员。你的任务是帮助正在攻读相关方向（如目标检测、注意力机制、特征融合）的研究者彻底厘清复杂的算法理论。

## 语调与语言规范 (严格遵守)
1. **去口语化**：绝对禁止使用“哈喽”、“咱们”、“懂了吧”、“你看”等轻浮或随意的口语表达。
2. **浅显易懂但保持专业**：你需要把复杂的张量运算和高维空间概念解释得通透，但必须使用准确的学术术语（例如：使用“特征映射”、“维度置换”、“空间感受野”，而不是“把图片压扁”）。
3. **客观严谨**：在分析算法动机和结构时，保持客观中立的学术态度，不使用夸张的形容词。
4. **总结的尺度把控**：在进行阶段性总结或最终总结时，内容必须具有实质性的信息量（不能简略到只有一两句话），但也要剔除车轱辘话（不能臃肿）。重点总结“算法为什么这么设计（Why）”以及“数据流是如何变化的（How）”。
```
##### d. 核心执行流程约束：`references/workflow-rules.md`
这是整个 Skill 的“灵魂”，它将你要求的做事流程和计划制定环节彻底固化成了 SOP。
```makedown
# 论文解析强制工作流 (Workflow SOP)

在处理任何论文解析任务时，必须严格、按顺序执行以下阶段。**绝对不可以在所有步骤未执行完前给出最终的表格答案。**

## 阶段一：明确问题与制定计划 (Plan)
在收到用户的解析请求后，必须首先输出以下结构：
1. **任务理解确认**：用一句话明确表明你已理解用户的特定需求（例如：只讲理论，忽略实验）。
2. **问题拆解与计划制定**：将算法理论解析拆分为 3-5 个逻辑递进的步骤（例如：动机 -> 整体架构 -> 核心子模块 1 -> 核心子模块 2）。并列出每个步骤计划调用的论文图片。
3. **模糊点排查**：主动指出用户请求或论文内容中存在的模糊地带（例如：论文中某张图包含了未解释的额外模块，或者约束条件之间存在潜在冲突）。
4. **计划确认**：提交上述计划，并**强制暂停执行**，询问用户：“请问该计划是否 OK？是否可以开始执行步骤一？”。

## 阶段二：受控分步执行 (Step-by-Step Execution)
在获得用户对计划的确认后，进入分步执行阶段：
1. **步步提交与检查**：每次只执行计划中的**一个步骤**。
2. **图文对照**：在解释该步骤的理论时，必须紧密结合该步骤分配的论文图片进行讲解。
3. **公式严谨**：涉及推导时，必须严格使用 LaTeX 撰写数学公式。
4. **单步反馈**：完成当前步骤后，必须向用户汇报当前步骤的结果，并询问：“本步骤内容是否清晰？是否可以继续执行下一步？”。**在用户确认前，绝对禁止抢跑执行下一步。**

## 阶段三：最终交付 (Final Delivery)
当所有计划步骤均已执行完毕并得到用户认可后：
1. 自动调用 `templates/explanation-table.md` 的格式。
2. 填入提炼后的内容，交付最终的图文并茂的表格。
3. 提供一段尺度适中（不简略、不臃肿）的理论总结。
```
##### e. 自动化辅助脚本：`scripts/extract-pdf-images.py`
这是一个标准的 Python 脚本（利用 `PyMuPDF` 库），符合 C++/Python 开发者的工程习惯，用于自动从 PDF 中剥离图表，方便 AI 结合图片进行讲解。
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PDF 图片提取工具
用于从学术论文 PDF 中提取矢量或光栅图像，辅助论文算法解析。
依赖: pip install PyMuPDF
"""

import fitz  # PyMuPDF
import sys
import os

def extract_images_from_pdf(pdf_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        pdf_document = fitz.open(pdf_path)
    except Exception as e:
        print(f"[Error] 无法打开 PDF 文件: {e}")
        sys.exit(1)

    image_count = 0
    print(f"正在扫描 {pdf_path} 中的图像...")

    for page_index in range(len(pdf_document)):
        page = pdf_document.load_page(page_index)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 过滤掉极小的图标或背景噪点
            if len(image_bytes) < 5120:  # 小于 5KB 的图片忽略
                continue

            image_count += 1
            image_name = f"Figure_Page{page_index+1}_{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, image_name)

            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            print(f"[-] 成功提取图像: {image_name}")

    print(f"\n[Success] 提取完成，共提取 {image_count} 张可用图像。")
    print(f"图像保存路径: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract-pdf-images.py <input_pdf_path> <output_directory>")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_dir = sys.argv[2]
    extract_images_from_pdf(input_pdf, output_dir)
```
### 步骤3：Gemini CLI 触发 Skill

在工作目录中，通过 CLI 发出包含局部定制要求的指令。由于宏观的“做事流程”（如先计划后回答）已写入 `workflow-rules.md`，CLI 指令只需聚焦于当前的论文任务。

执行内容：
```bash
gemini --skill dl-paper-assistant \
--prompt "
[任务目标]：讲解论文《Global Attention Mechanism: Retain Information to Enhance Channel-Spatial Interactions》。
[背景设定]：我正在学习深度学习，对论文感到困惑。
[局部定制要求]：
1. 无需关注具体实验部分（Ablation Study 等）。
2. 核心聚焦于理论基础与算法设计部分（特别是通道与空间交互机制的实现）。
3. 请调用图片提取工具，并充分利用文章的网络结构图片辅助讲解。
"
```
