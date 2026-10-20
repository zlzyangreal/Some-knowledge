# Spec: 《[等待提取：论文名称]》 代码复现与全维评估

## 1. Hardware Feasibility Assessment (硬件可行性评估)

在开始复现代码或规划架构之前，**必须**首先评估目标论文算法及源码在本地硬件上的运行可行性。

- **本地硬件基准**:
    - **GPU**: NVIDIA RTX 4070 Ti Super, 16 GB VRAM
    - **CPU**: Intel Core i7-13700K
    - **Memory**: 32 GB RAM
- **评估强制要求**: 
    - 结合目标模型的网络复杂度（参数量 #Params、计算量 GFLOPs），明确说明 16 GB 显存是否足以支撑该模型的**完整训练**与**推理**。
    - 若评估存在显存溢出 (OOM) 风险，必须在输出代码前提出具体的显存优化策略（如：推荐的最大 Batch Size、是否需要开启 AMP 混合精度训练、梯度累加步数建议等）。
    - 若该算法极度庞大（如某些超大视觉模型），完全超出单卡 16GB 显存的微调极限，需立即中止代码生成并说明不可行的原因。

## 2. Goal (目标)

基于提供的论文 Markdown 解析文件及源码（若有），复现其核心目标检测算法，并针对特定工业场景进行鲁棒性评估。

- **资料来源**: [等待输入：Markdown 文本及 GitHub 链接]
    
- **源码优先 (In-place Modification)**: 若有可靠开源仓库，**必须直接通过 `git clone` 拉取原始工程并在其原有架构上直接修改**。严禁以源码为蓝本从头重写或生成一套全新代码。任何被修改的核心文件头部需保留原 License 并注明原始 Git 链接及 commit。
    
- **核心任务**: 剥离公开数据集绑定，改造为纯算法实现，针对工业目标物体（`xia_liao`）场景无缝对接本地私有 YOLO 格式数据集，并完成高难度场景的性能压测。
    

## 3. Tech Stack (技术栈与环境)

- **环境基准 (强制要求)**:
    
    - **环境版本限制**: 必须明确声明支持的**最低 Python 版本**与**最低 PyTorch 版本**。
        
    - **PyTorch 专属声明**: 禁止输出直接安装 PyTorch 的 `pip` 指令，必须用醒目文本引导至 PyTorch 官网 (https://pytorch.org/) 根据本地 CUDA 版本自行获取安装命令。
        
    - 其他第三方依赖（如 `ultralytics`, `opencv-python`, `thop` 等）需提供一键执行的 `pip install` 指令。
        
- **核心框架**: PyTorch, Ultralytics, OpenCV, NumPy
    
- **运行环境**: Linux (Ubuntu)
    
- **约束条件**: 禁用自动下载数据集脚本。核心逻辑必须使用类型提示 (Type Hints)。
    

## 4. Data Structure & I/O (数据流与接口)

- **训练输入**: 解析 YAML 读取私有数据集。
    
    ```yaml
    path: /home/alphafox/Desktop/Data/Labels_Data/Clean_data/xia_liao/comparison_data/
    train: ./train/images
    val: ./val/images
    test: ./test
    names:
      0: xia_liao
    nc: 1
    ```
    
- **测试输入 (硬编码预设)**:
    
    - 遮挡困难集 (COVERED): `/home/alphafox/Desktop/Data/Labels_Data/Clean_data/xia_liao/test_for_parper/covered`
        
    - 烟雾困难集 (SMOG): `/home/alphafox/Desktop/Data/Labels_Data/Clean_data/xia_liao/test_for_parper/smog`
        
    - _(注：以上两数据集均包含 `images` 和 `labels` 子文件夹，标签为标准 YOLO txt 格式)_
        
- **输出**:
    
    - 训练阶段：Loss 日志、验证指标、权重文件（`.pt`）。
        
    - 推理阶段：可视化 Bounding Box 图片及坐标结果。
        
    - 测试阶段：包含模型复杂度及三大数据集全维度评估指标的 **`.txt` 结果汇总文件**。
        

## 5. Pipeline (核心执行逻辑)

1. **配置解析**: YAML 读取并自动拼接完整训练数据流路径。
    
2. **数据加载**: 编写适配的 Dataset/DataLoader，确保图片与标签严格对应校验。
    
3. **模型构建与规模选择**: 基于引入的源码或解析文件，实例化目标网络结构。
    - **模型尺度对齐 (强制约束)**: 严禁强行增删网络层以凑齐特定参数量。若目标源码将模型划分了不同的尺度版本（例如 N/S/M/L 变体），请自动选择参数量和计算量最贴近 **260 layers, 2,504,190 parameters, 5.8 GFLOPs** 这一基准的版本进行复现；若源码不区分尺度版本，则直接使用其默认网络配置即可。
    
4. **架构解耦 (Core vs. Run)**: 严格区分核心计算逻辑与外层调用逻辑：
    
    - **主逻辑层 (Core Modules)**: 核心算法与执行路径需封装在独立的模块中（如 `train.py`, `eval.py` 等）。作为“引擎”，它们只负责接收参数并执行张量计算、模型训练等底层任务，不包含任何环境路由逻辑。
        
    - **传参启动层 (`run.py`)**: 必须提供一个独立于核心逻辑的顶层控制台脚本。作为“方向盘”，它本身不包含复杂的算法运算。**禁止使用 `argparse` 命令行传参**。必须在文件内部通过直观的全局变量（如 `MODE = 'train'`）来切换以下三大功能，并拉起对应的底层模块：
        
        - **Mode 1: Train (训练)**: 执行前向传播、Loss 计算、反向传播及常规验证，保存 `.pt` 权重。
            
        - **Mode 2: Inference (推理)**: 加载训练好的 `.pt` 文件，对指定的图片/视频流进行检测，输出可视化结果。
            
        - **Mode 3: Test/Eval (参数与指标压测)**:
            
            - 读取 `.pt` 权重，输出模型参数量 (**#Params**) 和计算复杂度 (**GFLOPs**)。
                
            - 依次在 YAML 预设 `test` 文件夹、`COVERED` 数据集、`SMOG` 数据集中运行静默推理。
                
            - 统计 `Precision`, `Recall`, `mAP50`, `mAP50-95`, `FPS`, `Latency`，并将所有结果格式化写入评估报告 `.txt` 中。
                

## 6. Implementation Rules (开发规范)

- **溯源清晰**: 采用或修改自开源仓库的代码，必须保留原 License，注释注明 Git 链接。
    
- **鲁棒性**: 文件 I/O 操作（读取 YAML/图片/标签、写入报告）必须包含异常捕获。
    
- **可读性**: 核心算法与评估计算逻辑必须添加详细的**中文注释**。
    
- **解耦性**: 除约定的测试困难集绝对路径外，严禁硬编码其他数据路径。
    

## 7. Execution Protocol (最终执行指令 - 致 AI)

- **严格限制**: 接收到输入后，**禁止**复述本 Spec、总结或输出占位符。
    
- **唯一动作 (严格按序执行)**:
    
    1. **输出硬件可行性评估**：简明扼要地给出 RTX 4070 Ti Super (16GB) 运行该算法的结论及显存分配建议。如果不可行，立刻停止后续步骤。
    2. 直接输出依赖说明（PyTorch 官网引导说明 + 其他依赖的 pip 指令）。
    3. 针对提供的 Git 仓库代码，按文件路径输出**具体的修改代码片段**（如指出在 `models/yolo.py` 的某行替换为什么代码），并输出负责测试评估的专门模块（若原库没有）。
    4. **强制输出 `run.py` 启动文件**（作为唯一控制台，必须是一个纯粹的变量传参脚本，通过修改文件内的配置变量即可一键调用底层模块执行 `train`/`inference`/`test`）。
        
- **配置锚定**: 代码默认无缝对接上述预设的 YAML 和所有绝对测试路径及单类别，不进行任何多余假设。