# AI Agent之间的调用

## 结论

`Voyager` 不是 `Markdown` 驱动的，而是：

- `Python` 负责总控
- `Prompt` 负责给模型下指令
- `LLM` 生成动作/代码
- 环境执行
- 再把结果反馈回来

`Markdown` 如果有用，通常只是：

- 存 prompt
- 存模式说明
- 存配置文档

真正的调用和切换还是靠代码。

## Voyager 的基本结构

可以粗看成 3 个 Agent：

1. `CurriculumAgent`
   负责决定下一步做什么
2. `ActionAgent`
   负责根据任务和状态生成动作
3. `CriticAgent`
   负责判断这一步是否成功

外面还有一个总控循环把它们串起来。

## 调用链

```text
State -> CurriculumAgent -> Task
Task + State -> ActionAgent -> Action/Code
Action -> Environment -> Result
Result -> CriticAgent -> Success/Fail
Success/Fail -> Next Round
```

所以本质上不是 `Agent A` 直接调 `Agent B`，而是 `Controller` 依次调用多个 Agent。

## 胆大模式 / 胆小模式怎么实现

本质不是换了两套 Agent，而是：

- 同一套 Agent 架构
- 两套不同策略
- 用一个 `mode` 控制 prompt 和规则

例如：

```python
mode = "bold"       # 胆大
mode = "cautious"   # 胆小
```

### 胆大模式

- 高收益优先
- 接受高风险
- 更愿意探索危险区域
- 更快推进任务

### 胆小模式

- 生存优先
- 尽量不死
- 规避高风险战斗
- 先补给、先准备、再行动

## 一般改哪几层

### 1. 任务规划层

改 `CurriculumAgent`：

- 胆大：优先高收益任务
- 胆小：优先生存和稳健任务

### 2. 动作生成层

改 `ActionAgent`：

- 胆大：更激进
- 胆小：更保守

### 3. 评价层

改 `CriticAgent`：

- 胆大：允许更高失败率
- 胆小：把“不死”也作为成功标准

## 如果是通过 Markdown 切换

那本质也不是 `Markdown` 在调 Agent，而是：

- `Markdown` 里写策略说明
- `Python` 读取 markdown
- 再拼进 prompt

例如：

```python
if mode == "bold":
    system_prompt = base_prompt + read_file("bold.md")
else:
    system_prompt = base_prompt + read_file("cautious.md")
```

所以：

- `Markdown` 是载体
- `Python` 是执行器

## 最简实现思路

```python
policy = MODE_CONFIG[mode]

task = curriculum_agent.propose_task(state, policy=policy)
action = action_agent.generate(state, task, policy=policy)
review = critic_agent.evaluate(task, result, policy=policy)
```

## 一句话记忆

> `Voyager` 是 `Python + Prompt + LLM + 环境反馈` 的 Agent 系统；“胆大 / 胆小模式”本质上是对同一套 Agent 的策略切换，不是换了两个完全不同的 Agent。
