# AI 智能体驱动的嵌入式上位机开发实战 —— 以 MaixCAM 为例

> 用 AI Agent + Skill + Rules 构建一套规范、高效、可复用的 C++ 上位机工程体系

---

## 目录

1. [概念框架：AI → Agent → Skill → Rules](#一概念框架a-i-agent--skill--rules)
2. [C++ 上位机 AI 辅助开发实践](#二c-上位机-ai-辅助开发实践)
3. [AI 驱动的 Git 版本管理规范](#三ai-驱动的-git-版本管理规范)
4. [个人工作流构建与持续优化](#四个人工作流构建与持续优化)

---

## 一、概念框架：AI → Agent → Skill → Rules

### 1.1 上位机开发的真实痛点

以 MaixCAM（SG2002 RISC-V + Linux）作为主控上位机为例，需要同时处理：

```
┌──────────────────────────────────────────────────────────────┐
│               MaixCAM 上位机技术领域                           │
├──────────────────────────────────────────────────────────────┤
│  AI 推理      │ YOLO 目标检测 + ByteTracker 多目标跟踪          │
│  图像处理     │ 摄像头采集、H.264 编码、WebRTC 推流              │
│  通信协议     │ UART 自定义二进制帧：[0xE1]...[0xF1] + CRC16    │
│  控制算法     │ FSM 状态机 + PID 电机控制 + IMU 姿态解算          │
│  资源管理     │ 串口/终端/编码器/文件句柄的生命周期管理             │
└──────────────────────────────────────────────────────────────┘
```

任何一个领域出问题都会导致系统不可靠——UART 帧解析错位则控制指令乱掉、RAII 缺失则资源泄漏累积到死机。AI Agent 的核心价值不是"写代码更快"，而是**弥合这些跨领域知识之间的鸿沟，同时用 Rules 守住质量底线**。

### 1.2 四层定义

| 层级          | 角色     | 在上位机项目中的实际表现                                               |
| ----------- | ------ | ---------------------------------------------------------- |
| **AI（LLM）** | 跨领域知识库 | 理解 YOLO 模型加载、UART 帧格式、PID 参数，串起多领域代码                       |
| **Agent**   | 工程执行者  | 读取项目文件、执行 `riscv64-unknown-linux-musl-g++` 编译、联网查文档、维护项目记忆 |
| **Skill**   | 可复用模块  | 特定领域的模板与模式，以 `.md` 文件存储，触发时自动注入上下文                         |
| **Rules**   | 强制约束   | 命名规范、RAII 强制、include 顺序、通信超时——违反就拒绝的硬规则                    |

### 1.3 Agent 在真实项目中的操作

以下来自 MaixCAM 上位机的实际开发过程：

**操作一：发现 RAII 封装机会**

```
Agent 读取 keyboard.cpp：
  → 发现 set_terminal_raw() 和 reset_terminal() 之间无异常保护
  → 指出如果在中间的代码抛出异常，终端将永久处于 raw 模式
  → 生成 terminal_guard.hpp + terminal_guard.cpp
  → 构造函数调用 set_terminal_raw()，析构函数调用 reset_terminal()
  → 异常安全自动保证
```

**操作二：封装识别与建议**

```
Agent 读取 config.hpp：
  → 识别 struct RecorderConfig 所有字段为 public
  → 外部代码可以直接绕过 validate() 方法修改字段
  → 建议改为 class，字段 private，通过 getter/setter 访问
  → Rule 强制：成员变量必须 private + 后缀 _
  → Skill 提供 getter/setter 四种类型模板
```

**操作三：协议帧分析**

```
Agent 检查 UART 协议帧解析代码：
  → 发现 PARAM1 字段以小端序直接读取
  → 但帧格式定义中 PARAM1 高位在前（大端序）
  → 指出需要增加字节序转换
  → 同步更新 version-compatibility.yaml 补充说明
```

### 1.4 Skill —— 沉淀可复用的工程模式

Skill 是一个标注了触发条件的 `.md` 文件。Agent 检测到匹配场景时自动加载。

**建议的 MaixCAM 上位机 Skill 体系**：

```
~/.workbuddy/skills/
├── cpp-oop-patterns/        ← class 骨架、四种类型 getter/setter、RAII 模板
├── uart-protocol-framing/   ← 帧头对齐、定长帧提取、超时重试、CRC16 校验
├── yolo-detection-pipeline/ ← YOLO 模型加载、NPU 推理、后处理、ByteTracker 对接
├── webrtc-streaming/        ← WebRTC 推流、信令握手、H.264 IDR 帧间隔
├── fsm-state-machine/       ← FSM 状态机模板、转移表、守护条件
├── pid-control/             ← PID 参数结构体、调参接口、抗积分饱和
└── cpp-testing-mock/        ← Mock 硬件设备、gtest 模板
```

**Skill 的执行机制**：当用户提到「封装 class」「写 getter/setter」时，Agent 自动加载 `cpp-oop-patterns`，将 class 骨架和四种类型模板注入上下文，生成的代码自然风格一致。

### 1.5 Rules —— 不可妥协的硬性约束

放在项目根目录 `.workbuddy/rules/rule.mdc`，**Agent 每次对话自动加载**：

```yaml
# C++ 上位机强制规则

naming:
  - 类名 PascalCase，方法 camelCase，成员变量 snake_case_
  - 所有成员变量必须 private + 后缀 _

resource:
  - 禁止裸 new/delete，统一使用 RAII 或 std::unique_ptr
  - 文件句柄、串口、终端模式、编码器会话等资源必须封装为 RAII 类

structure:
  - 头文件 include 顺序：项目自身 → 框架 → 标准库
  - .hpp 放声明和简单 inline 方法，.cpp 放复杂实现

communication:
  - 所有串口/网络请求必须设置超时（默认 5s）
  - 通信失败重试 3 次，指数退避（1s/2s/4s）
```

### 1.6 四层协同的完整流程

```
用户：「把 RecorderConfig 从 struct 改成 class，加 getter/setter」

  → Agent 读取 simple_recorder.hpp                  [Agent 上下文获取]
  → Skill cpp-oop-patterns 注入 getter/setter 模板   [Skill 注入模式]
  → Rule 强制：字段 private + _ 后缀、禁止裸指针      [Rules 约束边界]
  → Agent 生成 solution/recorder_config.hpp          [AI 生成代码]
  → 人工审查：setter 中增加的校验逻辑是否合理           [人确认关键决策]
```

**关键洞察**：用户只需描述"想做什么"，而"怎么做才规范"由 Skill 和 Rules 自动保证。

---

## 二、C++ 上位机 AI 辅助开发实践

### 2.1 代码生成：渐进式三层推进

直接让 AI 生成完整类容易产生"能跑但不可维护"的代码。实践中有效的方法是分层推进：

**阶段一 —— 模板驱动**

提供一个字段的 getter/setter 作为模板，其余字段按相同模式批量生成。例如先让 AI 生成 `width_` 字段的完整 getter（含 `const` 版本）和 setter（含校验），确认风格正确后，说一句「其余字段照此模式」，AI 批量生成。

**阶段二 —— 三层代码递进审查**

Agent 对生成的代码执行三层递进审查：

| 层     | 检查内容              | 示例                                               |
| ------ | ----------------- | ------------------------------------------------ |
| **语法** | 变量名、类型匹配、const 位置 | `std::string x_ = val;` → setter 参数名遮蔽，实际定义了局部变量 |
| **模式** | RAII 缺失、资源泄漏、对子断裂 | `fopen()` 和 `fclose()` 中间无异常保护，一个提前 return 就泄漏   |
| **架构** | 封装不足、耦合过紧、接口混乱    | struct 全 public 字段，外部代码绕过 `validate()` 直接修改      |

**阶段三 —— 抽象沉淀**

审查通过的模式提取为 Skill 存�� `cpp-oop-patterns`，模板代码归档到 `mylib/` 可复用目录。下次同类需求直接复用，无需再走完整的生成→审查→修正流程。

### 2.2 代码审查：RAII「对子」扫描

Agent 对上位机代码最实用的一项审查能力是**检测「资源对子」**——成对出现但可能因异常或提前 return 而断裂的获取/释放操作。

Agent 扫描整个项目，自动标记所有对子，评估中间代码的异常安全性：

| 对子模式 | 原始代码                                      | 封装后                     |
| ---- | ----------------------------------------- | ----------------------- |
| 终端模式 | `set_terminal_raw()` / `reset_terminal()` | `TerminalGuard` RAII 类  |
| 串口资源 | `serial.open()` / `serial.close()`        | `SerialGuard` RAII 类    |
| 文件句柄 | `fopen()` / `fclose()`                    | `FileWriter` RAII 类     |
| 线程锁  | `mutex.lock()` / `mutex.unlock()`         | `std::lock_guard`       |
| 编码器  | `venc_start()` / `venc_stop()`            | `EncoderSession` RAII 类 |

**实际效果**：一次对子扫描在一个 5000 行的项目中发现了 3 处潜在的资源泄漏——两处在异常路径上、一处在 `goto` 跳转中。封装为 RAII 后，这些问题自动消除。

### 2.3 编译闭环：RISC-V 交叉编译 + 实机验证

MaixCAM 使用 RISC-V 交叉编译工具链（`riscv64-unknown-linux-musl-g++`），单次编译耗时较长（30s-2min）。每一次无效编译都是对开发节奏的打断。Agent 前置过滤可大幅减少浪费：

```
代码修改
  │
  ▼
Agent 静态审查                           ← 0 成本，秒级
  ├── 语法检查（类型匹配、缺少分号等）
  ├── include 依赖完整性
  └── Rule 合规（命名、RAII、裸指针）
  │
  ▼
人工确认                                 ← 关键决策点
  │
  ▼
cmake + make (riscv64 交叉编译)           ← 30s-2min
  │
  ├── 编译错误 → Agent 读取错误输出 → 诊断修复 → 重新编译
  │
  ▼
scp 部署到 MaixCAM 设备                   ← 网络传输
  │
  ▼
实机运行验证                              ← 最终确认
```

**核心收益**：Agent 前置审查能将"编译-报错-修-再编译"的循环从 3-5 轮压缩到 1-2 轮。语法级错误在提交编译前就被拦截了。

### 2.4 UART 协议帧解析 —— AI 辅助的通信层实现

上位机与 STM32 下位机之间的 UART 通信使用自定义二进制帧格式，这是一个经典的"看起来简单写起来坑多"的领域：

```c
// 协议 v1：8 字节定长帧
// [0xE1][type][state][p1H][p1L][p2H][p2L][0xF1]
//
// 协议 v2：10 字节（增加 CRC16）
// [0xE1][type][state][p1H][p1L][p2H][p2L][crcH][crcL]

// AI 生成的帧解析器骨架（基于 uart-protocol-framing Skill）
class UartFrameParser {
public:
    // 喂入原始字节流，返回解析完成的帧（可能为空）
    std::optional<Frame> feed(uint8_t byte);
    
private:
    enum class State {
        WAIT_HEADER,    // 等待 0xE1
        READ_BODY,      // 读取固定长度
        CHECK_CRC,      // 验证 CRC16（v2 协议）
    };
    
    State state_ = State::WAIT_HEADER;
    std::vector<uint8_t> buffer_;
    size_t expected_len_ = 8;  // v1 默认 8 字节
    
    // 校验和预计错误次数，连续超过阈值则触发告警
    int consecutive_crc_errors_ = 0;
    static constexpr int MAX_CONSECUTIVE_ERRORS = 10;
};
```

Agent 生成的代码会自动处理：帧头误识别（在数据中遇到 `0xE1`）、超时清理（超过 500ms 未收到完整帧则丢弃缓冲区）、CRC 错误计数、触发告警。

### 2.5 状态机与 PID —— AI 辅助的控制层实现

MaixCAM 上位机使用 FSM 状态机管理整体行为，PID 控制执行机构的运动：

```cpp
// AI 基于 fsm-state-machine Skill 生成的状态机骨架
enum class AppState {
    IDLE,           // 待机，等待指令
    SCANNING,      // 扫描/巡航中
    TRACKING,      // 已发现目标，正在跟踪
    PROCESSING,    // 处理中（如执行特定任务）
    ERROR,         // 异常状态（通信丢失/执行故障）
};

// 状态转移表（AI 自动检查：是否有不可达状态？是否有死锁环路？）
const StateTransition transitions[] = {
    {IDLE,        CMD_START,        SCANNING},
    {SCANNING,    TARGET_DETECTED,  TRACKING},
    {SCANNING,    CMD_STOP,         IDLE},
    {TRACKING,    TARGET_LOST,      SCANNING},
    {TRACKING,    TARGET_CONFIRMED, PROCESSING},
    {PROCESSING,  TASK_DONE,        SCANNING},
    {ANY,         COMM_LOST,        ERROR},
    {ERROR,       CMD_RESET,        IDLE},
};
```

Agent 的价值在于：**自动检查状态转移表的完整性**——有没有不可达的状态？有没有状态下缺少超时保护？有没有转移条件重复冲突？

---

## 三、AI 驱动的 Git 版本管理规范

### 3.1 Commit 类型扩展

在 Conventional Commits 基础上，增加上位机专属类型，scope 精确到模块：

| 类型         | scope 示例         | 实际 commit                                  |
| ---------- | ---------------- | ------------------------------------------ |
| `feat`     | `config`         | `feat(config): Config 类封装，支持 YAML 热加载`     |
| `refactor` | `driver`, `raii` | `refactor(raii): TerminalGuard 封装替代裸函数对`   |
| `fix`      | `webrtc`, `uart` | `fix(webrtc): H.264 IDR 帧间隔调整解决 I 帧丢失`     |
| `perf`     | `yolo`, `pid`    | `perf(yolo): ByteTracker 匹配阈值调优，跟踪率提升 15%` |
| `docs`     | `skill`          | `docs(skill): 新增 RAII 对子扫描模式`              |

AI 参与的 commit 在 body 中标注 `Ai-assisted: true`。

### 3.2 上位机+下位机双端版本锁定

嵌入式项目的关键约束：**上位机协议版本与下位机固件版本必须配套**，否则帧解析直接失败。

```yaml
# .context/version-compatibility.yaml
# Agent 每次对话自动加载，作为代码生成的约束条件

compatibility:
  - app: "v3.0"
    min_firmware: "STM32_v2.1"
    protocol_version: 1
    frame_format: "8字节定长 [0xE1][type][state][p1H][p1L][p2H][p2L][0xF1]"
    
  - app: "v3.1"
    min_firmware: "STM32_v2.4"
    protocol_version: 2
    frame_format: "10字节 [0xE1][type][state][p1H][p1L][p2H][p2L][crcH][crcL]"
    breaking_changes:
      - "帧尾从固定 0xF1 改为双字节 CRC16 校验"
      - "IMU 上报频率从 10Hz 提升至 50Hz"
```

固件升级改变了通信协议，**第一步永远是更新这份兼容表**。Agent 在生成或修改 UART 代码时自动读取当前版本对应的帧格式。

### 3.3 PR 模板

```markdown
## 变更概述
[AI 根据 diff 自动生成]

## 变更类型
- [ ] 新功能  - [ ] 驱动重构  - [ ] 通信层变更
- [ ] Bug修复 - [ ] 性能优化  - [ ] 配置/文档

## 影响模块
- [ ] YOLO      - [ ] ByteTracker  - [ ] FSM 状态机
- [ ] UART      - [ ] WebRTC       - [ ] SimpleRecorder
- [ ] PID       - [ ] Config

## 固件兼容性
- 最低固件版本：___
- Breaking Change：[是/否]
- 如 Breaking Change：是否已更新 version-compatibility.yaml

## AI 参与声明
- [ ] 代码 AI 生成，已人工审查
- [ ] AI 辅助 Review，代码全手写
- [ ] 无 AI 参与

## 测试
- [ ] RISC-V 交叉编译通过
- [ ] MaixCAM 实机验证通过
```

### 3.4 CI/CD 流水线

```yaml
jobs:
  lint:        # clang-format 代码风格 + clang-tidy 静态分析
  build:       # riscv64-unknown-linux-musl-g++ 交叉编译
  ai-review:   # Agent 专项检查——裸指针、RAII 缺失、UART 超时
  package:     # 产出 wj_app_v3 + config.yaml 固件分发包
```

### 3.5 `.gitignore` 关键条目

```gitignore
.workbuddy/memory/     # 个人日志，不提交
*.mud *.cvimodel       # AI 模型文件（体积大，单独存储）
build/ *.o             # 编译产物
```

---

## 四、个人工作流构建与持续优化

### 4.1 项目目录结构 —— Agent 的"认知基础"

```
maixcam-upper/
├── main/
│   ├── include/           ← .hpp（按模块分：core/ drivers/ fsm/ webrtc/ utils/）
│   └── src/               ← .cpp
├── mylib/                 ← 可复用库（YOLO, SimpleRecorder, UartComm）
├── test/                  ← 单元测试（gtest）
├── .context/              ← Agent 上下文（每次对话自动加载）
│   ├── ARCHITECTURE.md                    ← 系统架构说明
│   ├── CODING_STANDARDS.md                ← 编码规范
│   └── version-compatibility.yaml         ← 版本兼容表
├── .workbuddy/
│   ├── rules/rule.mdc     ← 硬性编码规则（Agent 每次对话自动加载）
│   └── memory/            ← 项目工作日志
├── solution/              ← AI 生成的方案代码（审查通过后合并到 main/）
├── config.yaml
└── CMakeLists.txt
```

**`.context/` 目录是 Agent 的「认知基础」**——设备 API 协议、系统架构、编码规范全部集中于此。Agent 每次对话首先读取这些文件，确保生成的代码与项目既有风格和协议约定一致。`solution/` 目录则是 AI 生成代码的"缓冲区"，审查通过后手动合并，确保每行进入主干代码的都经过人眼确认。

### 4.2 可复用库的构建策略

`mylib/` 项目的生命力来自持续抽象。以下是建议的构建路线：

```
阶段一（第 1-2 周）：基础模式沉淀
  - UartComm：      串口收发 + 帧解析 + 超时重连
  - ConfigManager：  YAML 加载 + 热重载 + 参数校验

阶段二（第 3-4 周）：核心算法组件化
  - YOLO：          NPU 模型加载 + 推理 + 后处理
  - ByteTracker：   多目标跟踪 + ID 分配 + 轨迹管理
  - SimpleRecorder：H.264 录像 + 分段存储 + 循环覆盖

阶段三（第 5-8 周）：控制与通信完善
  - PIDController： 增量式/位置式 PID + 参数在线调优
  - FsmEngine：      通用状态机引擎 + 转移表验证
  - WebRTCStreamer： 推流封装 + 自适应码率
```

每个模块从具体项目中提取，经过 2-3 次实战验证后，整理接口文档并移到 `mylib/` 下。新项目直接引入，不再重复造轮子。

### 4.3 实操案例：新增 PID 控制模块

> 💡 **阅读提示**：以下为完整闭环演示（09:00-09:25），若时间紧张可直接跳至文末 **"三个关键的持续动作"** 查看精华总结。

```
时间线                         人机分工
───────────────────────────────────────────────────────────

09:00  开发者描述需求                     👤 人
       "执行机构需要 PID 控制，先实现位置式 PID，
        参数从 config.yaml 读取，支持在线调参。"

09:02  Agent 分析项目上下文               🤖 AI
       - 读取 .context/ARCHITECTURE.md
       - 读取 .workbuddy/rules/rule.mdc
       - 读取现有 config.yaml 结构
       - 检查 mylib/ 是否已有可复用组件

09:04  Agent 提出设计方案                  🤖 AI
       "建议新增 mylib/PIDController：
        - 位置式 PID 结构体，字段 private + _ 后缀
        - 从 config.yaml 的 [pid] 段加载参数
        - 提供在线调参接口，通过 UART 指令下发
        - 实现抗积分饱和 (clamping)
        - 集成到控制线程中（独立于摄像头主循环）"

09:06  开发者审查方案                     👤 人
       "方案合理。补充：
        - 增加死区（deadband），避免小误差振荡
        - 输出需要限幅（有机械限位）"

09:08  Agent 生成代码                      🤖 AI
       - pid_controller.hpp + pid_controller.cpp
       - 从 pid-control Skill 注入抗积分饱和模式
       - 从 config.yaml 读取 [pid] 段（复用 ConfigManager）
       - 修改 motor_control.cpp 集成 PID
       - test/pid_controller_test.cpp（单元测试）

09:12  开发者审查代码                     👤 人
       "constructor 中 config 参数应该 const 引用，
        避免不必要的拷贝。"
       同时也发现：输出限幅使用硬编码 ±90°，
       建议改为 #define 或 config 可配

09:14  Agent 修复                          🤖 AI
       - 参数改为 const Config&
       - 输出限幅改为 config 配置项

09:16  编译 + scp 部署                    👤 人 + 🤖 AI
       riscv64 交叉编译通过 → scp 上传
       → 实机运行 → 响应延迟偏大

       Agent 分析：默认 P=0.5 偏保守，
       且积分分离阈值设置不当导致积分项贡献不足。
       建议 P=1.2, I=0.3, 分离阈值调大。

09:22  调参后重新测试                     👤 人
       跟踪流畅，无振荡，无超调。
       通过 UART 下发 PID 参数，在线调参功能正常。

09:25  提交 PR                            👤 人
       Agent 生成 PR 描述                  🤖 AI
       标注：无 Breaking Change，固件兼容性不变
```

### 4.4 持续迭代闭环

```
需求提出
  │
  ▼
Agent 分析 .context/ + .workbuddy/rules/   ← 注入项目知识和约束
  │
  ▼
Skill 提供领域模板                          ← 复用验证的模式
  │
  ▼
生成代码到 solution/                        ← 缓冲层，未直接进入主干
  │
  ▼
人工审查（语法 → 模式 → 架构 三层）          ← 核心质量把控
  │
  ▼
RISC-V 交叉编译 + scp 部署 + 实机验证         ← 真机才是最终答案
  │
  ▼
经验沉淀：通用模式 → Skill / 可复用模块 → mylib/
```

**三个关键的持续动作**：

🔑 **每次固件协议变更，第一时间更新 `version-compatibility.yaml`** ——这是 Agent 生成通信代码的约束来源。（口诀：**协议变，表先动**）

🔑 **每次发现通用模式，提取为 Skill** ——RAII 对子扫描、UART 帧解析状态机、PID 抗积分饱和，这些不应该写两遍。（口诀：**模式即资产**）

🔑 **每次遇到新的边界情况，补充 Rules** ——命名约束、裸指针禁令、超时强制——规则越完善，Agent 越可靠。（口诀：**坑过即规则**）

项目迭代越多，上下文越丰富，Agent 越"懂"这个项目。三个月后回头看，那些反复踩过的坑不再出现，不是因为记性好，而是因为 Skill 和 Rules 替每个人都记住了。

---

## 结语

MaixCAM 上位机开发的特点决定了——它不是任何一个单一技术栈的深度应用，而是**AI 推理、通信协议、控制算法、资源管理**四个领域的交叉地带。传统的开发方式是每个领域各查各的文档、各写各的代码、出了问题各调试各的模块。AI Agent 的引入改变的是**协同方式**：

1. **跨领域知识不再是瓶颈**：YOLO 的模型加载、UART 的帧格式、PID 的参数——AI 可以同时提供三个领域的代码，你只需要确认接口对齐和逻辑正确

2. **规范不再是"自觉"**：RAII 对子扫描确保资源不泄漏、三层代码审查覆盖语法-模式-架构、Rules 锁定命名和资源管理方式——质量是设计出来的，不是检查出来的

3. **经验可以传递**：每个踩过的坑（字节序忘转、IDR 帧间隔不对、积分饱和）不再是老工程师脑子里的隐性知识，而是结构化的 Skill 和 Rules。下一次开发，Agent 自动帮你绕过去

4. **编译-部署-验证循环加速**：Agent 前置审查过滤低级错误，交叉编译不再被反复的"少分号多括号"打断；scp 部署 + 实机验证的时间真正用在了有价值的调试上

**最实用的起步建议**：下周开始，选一个明确的小任务（比如把一个 struct 改成 class、给串口初始化加个超时），让 Agent 根据 Rules 和 Skill 生成代码，你负责审查。如果风格对了，扩展到下一个任务。如果发现模式可以复用，写到 Skill 里。一个月后回看，你会发现——不再是你在写代码，而是你在确认 Agent 写的代码是否符合你的标准。

**别让你的经验，只锁在你一个人的脑子里。**

工具在进化，开发方式也在进化。当AI能替你记住所有规范、扫掉所有低级错误时，工程师真正的价值，终于从枯燥的"垒代码"回归到了"解难题"。

试一试吧。下周选一个最头疼的小模块，交给Agent，你只负责按"确认键"。**很快你会发现，你不是在学怎么用AI，你是在重新定义自己作为工程师的上限。**

📎 **文末福利**：关注本公众号，后台回复关键词 **`MaixCAM规范`**，即可获取文中提到的 `rule.mdc` 和 `version-compatibility.yaml` 空白模板（可直接粘贴到你的项目中落地使用）。

---

*本文基于 Sipeed MaixCAM (SG2002 RISC-V) + C++/MaixCDK + WorkBuddy Agent 的真实工程实践撰写。技术栈和工具链随版本演进，建议每季度回顾并更新 Skill 与 Rules 规则。*
