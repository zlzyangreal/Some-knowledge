# Spec: 《[等待提取：论文名称]》 追踪代码复现与全维评估 (Spec-reproduction-of-the-sort-code)

## 1. Hardware Feasibility Assessment (硬件可行性评估)

在开始复现代码或规划架构之前，**必须**首先评估目标论文算法及源码在本地硬件上的运行可行性。

- **本地硬件基准**:
    - **GPU**: NVIDIA RTX 4070 Ti Super, 16 GB VRAM
    - **CPU**: Intel Core i7-13700K
    - **Memory**: 32 GB RAM
- **评估强制要求**: 
    - 结合目标模型的网络复杂度（参数量 #Params、计算量 GFLOPs），明确说明 16 GB 显存是否足以支撑该多目标追踪 (MOT) 模型的**推理与运行**。
    - **追踪器专属评估**：由于追踪算法（如 SORT、DeepSORT、OS-SORT 等）通常包含多个子模块，必须评估其运行特征提取网络（如 ReID 外观模型）或目标检测器时的显存占用峰值。
    - 若评估存在显存溢出 (OOM) 风险，必须在输出代码前提出具体的显存优化策略（如限制视频读取缓存、调整检测器/特征提取层级等）。
    - 若该算法极度庞大，完全超出单卡 16GB 显存的运行极限，需立即中止代码生成并说明原因。

## 2. Goal (目标)

基于提供的**论文 Markdown 解析文件**及源码（若有），复现其核心多目标追踪 (MOT) 算法，并针对特定工业场景进行鲁棒性评估。

- **资料来源**: [等待输入：Markdown 文本及 GitHub 链接]
    
- **源码优先 (In-place Modification)**: 若有可靠开源仓库，**必须直接通过 `git clone` 拉取原始工程并在其原有架构上直接修改**。严禁从头重写。任何被修改的核心文件头部需保留原 License 并注明原始 Git 链接及 commit。
    
- **核心任务**: 剥离公开数据集绑定，改造为纯算法实现。对接本地私有视频数据，使用用户手动提供的本地权重（`.pt` 或 `.pth`），完成高难度场景（遮挡、烟雾）的追踪性能压测。
    

## 3. Tech Stack (技术栈与环境)

- **环境基准 (强制要求)**:
    - **环境版本限制**: 必须明确声明支持的**最低 Python 版本**与**最低 PyTorch 版本**。
    - **PyTorch 专属声明**: 禁止输出直接安装 PyTorch 的 `pip` 指令，必须用醒目文本引导至 PyTorch 官网自行获取安装命令。
    - 其他第三方依赖（如 `ultralytics`, `opencv-python`, `motmetrics` 等）需提供一键执行的 `pip install` 指令。
        
- **核心框架**: PyTorch, OpenCV, NumPy, 追踪评估计算库
- **运行环境**: Linux (Ubuntu)
- **约束条件**: 禁用自动下载数据集脚本。核心逻辑必须使用类型提示 (Type Hints)。
    

## 4. Data Structure & I/O (数据流与接口)

- **权重输入 (强制要求)**: **模型权重文件（`.pt` 或 `.pth`）由用户手动提供。** 必须在顶层脚本 `run.py` 中留出清晰的绝对路径变量 `WEIGHT_PATH` 供用户填写。

- **MOT 标签详细格式规范 (MOT Challenge Format)**:
    测试集提供的 `.txt` 标签严格遵循 MOT Challenge 标准。每行代表一个目标在某一帧的实例：
    `frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility, unused`
    
    **数据示例：**
    `1,1,1099.559,112.775,236.828,831.718,1.0,0,1,1`

- **测试输入路径 (硬编码预设)**:
    - **遮挡困难集 (COVERED)**: 
        - 视频源: `/home/alphafox/Desktop/Code/Yolo26/ultralytics/Data_mv/Covered/test_data`
        - MOT 标签: `/home/alphafox/Desktop/Data/Labels_Data/Clean_data/xia_liao/test_for_parper/mot_labels/covered`
    - **烟雾困难集 (SMOG)**: 
        - 视频源: `/home/alphafox/Desktop/Code/Yolo26/ultralytics/Data_mv/smog`
        - MOT 标签: `/home/alphafox/Desktop/Data/Labels_Data/Clean_data/xia_liao/test_for_parper/mot_labels/smog`

- **输出**:
    - 推理阶段：输出带 ID 和 Box 的可视化视频。
    - 测试阶段：生成包含 IDs, MOTA, AssA, ML, FPS 等指标的 `.txt` 评估报告。


## 5. Pipeline (核心执行逻辑)

1. **配置解析**: 自动拼接视频路径与对应的 MOT 标签文件路径。
    
2. **数据加载**: 实现 MOT 格式解析器，将 `.txt` 标签读取为张量或结构化对象，以便与追踪预测结果进行逐帧对齐。
    
3. **模型识别与配置说明 (强制约束)**: 
    - **识别**：基于提供的**论文 Markdown 解析文件**，必须明确指出论文实验所用的具体模型名称及尺度版本（如：`yolox_m` + `ByteTrack` 配置）。
    - **对接**：AI 无需编写复杂的模型构建逻辑，只需输出该模型所需的权重格式说明。用户将在 `run.py` 中通过 `WEIGHT_PATH` 变量直接链入本地的 `.pt` 或 `.pth` 文件。

4. **架构解耦 (Core vs. Run)**: 
    - **主逻辑层 (Core Modules)**: 追踪器的匹配逻辑与评估计算逻辑需保持纯净，不包含文件路径硬编码。
    - **传参启动层 (`run.py`)**: 必须提供一个独立于核心逻辑的顶层控制台脚本。**禁止使用 `argparse`**。
        - 变量：`MODE = 'inference'` (可选 'inference' 或 'eval'), `WEIGHT_PATH = '...'`。
        - **Mode 1: Inference (可行性证实)**: 加载用户权重，对指定视频跑通追踪流程，保存带 ID 的可视化视频，证实代码在本地成功复现。
        - **Mode 2: Eval (指标压测)**: 在 `COVERED` 和 `SMOG` 数据集上执行静默推理。调用评估库对比预测轨迹与 GT 标签，计算并输出指标：**IDs, MOTA, AssA, ML, FPS**。


## 6. Implementation Rules (开发规范)

- **溯源清晰**: 必须保留原始 License 和 Git 链接。
- **鲁棒性**: 对视频读取和标签解析必须进行异常捕获。
- **可读性**: 核心追踪算法（如代价矩阵计算、卡尔曼滤波更新）与评估逻辑必须添加详细的**中文注释**。
- **解耦性**: 严禁在底层算法模块中硬编码数据路径。


## 7. Execution Protocol (最终执行指令 - 致 AI)

- **严格限制**: **禁止**复述本 Spec 或输出总结。
    
- **唯一动作 (严格按序执行)**:
    1. **输出硬件可行性评估**：评估 RTX 4070 Ti Super (16GB) 运行该追踪算法的可行性。
    2. **论文模型识别**：根据**论文 Markdown 解析文件**附件，识别论文实验中使用的核心模型变体，明确告知用户需要准备哪种架构的权重文件。
    3. **输出环境依赖**：PyTorch 官网引导 + 其他依赖 pip 指令。
    4. **输出核心修改代码片段**：针对提供的仓库，输出适配本地视频流和 MOT 标签解析的修改方案。
    5. **强制输出 `run.py` 启动文件**：纯变量传参机制，一键调用底层执行推理证实或指标压测。
        
- **配置锚定**: 代码默认对接预设路径及上述定义的 MOT Challenge 标签格式。