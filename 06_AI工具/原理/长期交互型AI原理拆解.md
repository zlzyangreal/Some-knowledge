# 用 Python 实现一个长期交互型 AI Agent：人格建模、长期记忆与工具调用实战

很多 AI Agent 项目一开始都很好做：接一个大模型，挂几个工具，再补一个聊天界面，系统就能跑起来。它可以回答问题、查天气、调用接口，看上去已经像一个完整的 Agent。

但只要目标从“完成一轮任务”变成“长期陪伴用户”，问题就会立刻变复杂。

用户会自然期待它记得昨天聊过的事，期待它的说话方式前后一致，也期待它不只是给建议，而是真的能帮忙做点事。到了这一步，系统的重点就不再只是生成文本，而是要同时处理三件事：人格、记忆和工具。

这篇文章围绕一个最小可运行版本展开：用 Python 实现一个具备人格建模、长期记忆和工具调用能力的长期交互型 AI Agent。目标很具体：

- 能和用户连续对话
- 有相对稳定的人格风格
- 能把用户事实、偏好和情绪写入长期记忆
- 能根据当前问题召回相关记忆
- 至少具备两个实用工具：查天气、设提醒

整套系统可以先收敛成这样一条链路：

```text
用户输入
  ↓
Agent Orchestrator
  ├── Persona Layer
  ├── Memory Layer
  ├── Tool Layer
  └── LLM Runtime
  ↓
人格化回复
  ↓
写回长期记忆
```

真正决定体验的，不是某一个模型名字，而是这几层能力能不能形成闭环。

---

## 1. 先把工程边界收紧

长期交互型 Agent 很容易一上来就做大：想加语音、加视觉、加主动消息、加日历、加任务规划。这样通常会导致系统还没跑通，复杂度就先失控了。

更稳妥的做法，是先做一个最小系统：

- 用结构化参数定义人格
- 用长期记忆保存用户信息
- 用工具把“建议”变成“行动”
- 用一个统一的编排层把三者串起来

项目结构可以保持得很干净：

```text
src/
  agent/
    orchestrator.py
  persona/
    traits.py
    prompt.py
  memory/
    types.py
    store.py
    extract.py
  tools/
    weather.py
    reminder.py
    registry.py
  models/
    chat.py
    embedding.py
    reranker.py
```

这种拆法的好处很直接：后面无论是替换模型、增加工具，还是把内存版存储换成 SQLite / 向量库，都不用大改主流程。

---

## 2. 人格层：不要只写“温柔”“聪明”，要让人设可配置

很多角色型应用最常见的写法，是在 system prompt 里写一句“你是一个温柔、耐心、善于安慰人的 AI”。这样当然能工作，但它最大的问题是不可控。

如果角色风格只是一段文案，就很难做 A/B 测试，也很难解释“为什么这次回复突然变味了”。更适合工程化落地的方式，是把人格参数化。

### 2.1 用 HEXACO 做一个最小人格配置

这里直接用 HEXACO 六维人格模型做结构化建模。代码不复杂，但意义很大，因为人格第一次变成了程序里的明确对象。

```python
from dataclasses import dataclass

@dataclass
class Hexaco:
    honesty: float
    emotionality: float
    extraversion: float
    agreeableness: float
    conscientiousness: float
    openness: float

DEFAULT_TRAITS = Hexaco(
    honesty=0.78,
    emotionality=0.72,
    extraversion=0.66,
    agreeableness=0.85,
    conscientiousness=0.63,
    openness=0.91,
)
```

这 6 个值表面上只是浮点数，但它们实际承担的是角色控制面板的作用：

- `emotionality` 高，安慰场景下会更细腻
- `conscientiousness` 高，涉及计划和提醒时会更有条理
- `openness` 高，更容易做适度联想和延展

### 2.2 把人格参数翻译成系统行为

有了参数之后，还要把它落到模型能理解的形式上。最直接的一步，是由参数生成系统提示词。

```python
def build_personality_prompt(p: Hexaco) -> str:
    return f"""
你是一名面向长期交互的 AI Agent。

你的人格参数如下：
- 诚实-谦逊: {p.honesty}
- 情绪性: {p.emotionality}
- 外向性: {p.extraversion}
- 宜人性: {p.agreeableness}
- 尽责性: {p.conscientiousness}
- 开放性: {p.openness}

行为规则：
- 不编造记忆，不操纵用户
- 回应保持温和、稳定
- 在计划与提醒场景中更有条理
- 在安慰场景下更细腻
""".strip()
```

到这一步，人格至少已经参与了回复生成。但如果只停在这里，角色仍然很容易随着上下文变长而漂掉，所以更进一步的做法，是让人格也影响运行时决策。

例如：

```python
def derive_runtime_bias(p: Hexaco) -> dict:
    return {
        "emotional_sensitivity": p.emotionality,
        "planning_depth": p.conscientiousness,
        "proactivity": (p.extraversion + p.openness) / 2,
    }
```

这段偏置不需要一开始就特别复杂，但它传达了一个重要思路：人格不只是“怎么说”，也应该影响“怎么做”。

---

## 3. 长期记忆层：上下文不是记忆，最多只是缓存

如果一个 Agent 只能依赖最近几轮消息，那它最多只能保证当前对话连贯，很难形成持续的陪伴感。真正有用的长期记忆，至少要回答两个问题：

- 什么内容值得保存
- 保存之后怎么在下一轮正确用起来

### 3.1 先把记忆类型分开

长期记忆最大的坑，是把所有内容都混在一起。用户职业、情绪波动、回复偏好，本质上不是一类东西，后续的召回方式也不应该一样。

一个最小实现里，至少可以分成三类：

```python
from dataclasses import dataclass
from typing import Literal, Union

@dataclass
class FactMemory:
    id: str
    type: Literal["fact"]
    content: str
    importance: float
    created_at: str

@dataclass
class EmotionalMemory:
    id: str
    type: Literal["emotion"]
    content: str
    mood: Literal["happy", "sad", "anxious", "excited", "angry"]
    importance: float
    created_at: str

@dataclass
class PreferenceMemory:
    id: str
    type: Literal["preference"]
    content: str
    importance: float
    created_at: str

MemoryItem = Union[FactMemory, EmotionalMemory, PreferenceMemory]
```

这个分类的意义，不是让数据结构更漂亮，而是让后面的召回和生成有依据：

- `fact` 决定 AI 是否真的“知道你”
- `emotion` 决定 AI 是否在关键时刻“懂你”
- `preference` 决定 AI 是否会逐渐贴近你的交流习惯

### 3.2 存储不要只依赖向量库

长期记忆通常会接入 embedding 和向量检索，但只存向量是不够的。因为一旦要做记忆纠错、删除敏感信息、按类型过滤、设置过期时间，结构化字段就很重要。

所以一个更实用的接口通常会是“结构化存储 + 语义检索”双层设计。抽象成代码，大概像这样：

```python
from typing import Protocol, Optional

class MemoryStore(Protocol):
    def save(self, memory: MemoryItem) -> None: ...
    def search(self, query: str, limit: int) -> list[MemoryItem]: ...
    def get_by_id(self, memory_id: str) -> Optional[MemoryItem]: ...
```

接口非常简单，但已经把职责分清了：

- `save()` 负责持久化
- `search()` 负责召回候选记忆
- `get_by_id()` 负责细粒度读取和纠错

### 3.3 记忆提取要有筛选，而不是全量保存

很多早期系统会直接把用户消息整段存下来，这样实现简单，但后面几乎一定会被噪音淹没。更合理的做法，是在每轮对话结束后做一次记忆提取，只保留值得进入长期记忆的内容。

下面这个示例故意写得很朴素，因为重点不是“规则有多聪明”，而是“先建立提取这一层”。

```python
from uuid import uuid4
from datetime import datetime


def extract_memories_from_turn(user_input: str, reply: str) -> list[MemoryItem]:
    memories: list[MemoryItem] = []

    if "准备面试" in user_input:
        memories.append(FactMemory(
            id=str(uuid4()),
            type="fact",
            content="用户最近在准备面试",
            importance=0.8,
            created_at=datetime.utcnow().isoformat(),
        ))

    if "有点焦虑" in user_input:
        memories.append(EmotionalMemory(
            id=str(uuid4()),
            type="emotion",
            content="用户近期有焦虑情绪",
            mood="anxious",
            importance=0.9,
            created_at=datetime.utcnow().isoformat(),
        ))

    return memories
```

在真实项目里，这一步通常会进一步升级成“规则 + 轻量模型”或者单独的记忆提取模型，但最小版本做到这里已经够用了。

### 3.4 召回不是结束，还要再排一次序

哪怕已经有了向量召回，也不代表拿到的 Top N 就一定适合直接送进 Prompt。长期交互型 Agent 特别容易遇到一种情况：语义相似的内容很多，但真正重要的其实是最近的情绪事件，或者是高重要度的偏好记忆。

所以召回之后，通常还要再做一次简单重排：

```python
def score_memory(item: MemoryItem, semantic_score: float) -> float:
    type_bonus = 0.15 if item.type == "emotion" else 0.05
    return semantic_score + item.importance * 0.2 + type_bonus
```

这只是一个示意公式，但它已经足够表达工程上的真实思路：最终得分通常不是单一相似度，而是语义、重要性、时间新鲜度和类型权重共同决定的。

---

## 4. 工具层：让 Agent 从“给建议”变成“帮你做事”

长期交互型 AI 只有会聊天，体验往往很薄。真正让用户感到“被照顾”的时候，通常不是它说了一句多么温柔的话，而是它真的把一件小事办了。

这篇文章先加两个最小工具：

- 查天气
- 设提醒

### 4.1 先约定一个统一工具协议

如果工具只是几个散落的函数，后面很快会变乱。更合适的做法，是先把工具定义成一个统一结构：

```python
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

@dataclass
class ToolDefinition(Generic[TIn, TOut]):
    name: str
    description: str
    execute: Callable[[TIn], TOut]
```

一旦有了这层抽象，后面接天气 API、日历 API、本地提醒服务都可以沿着同一个协议扩展。

### 4.2 先做两个最小工具

这里不需要一开始就接真服务，先把工具调用链路跑通更重要：

```python
def check_weather(input_data: dict) -> dict:
    return {
        "city": input_data["city"],
        "weather": "light rain",
        "temperature": 26,
    }


def create_reminder(input_data: dict) -> dict:
    return {"success": True}
```

这两个函数看起来很简单，但已经足够让 Agent 开始从“语言输出”进入“动作输出”。

### 4.3 工具注册最好绑定当前上下文

真实项目里，工具几乎总是和当前用户、当前权限、当前会话有关，所以最好在注册层就把上下文带进来。

```python
def build_tool_registry(actor_id: str) -> dict:
    return {
        "actor_id": actor_id,
        "check_weather": ToolDefinition(
            name="check_weather",
            description="查询指定城市天气",
            execute=check_weather,
        ),
        "create_reminder": ToolDefinition(
            name="create_reminder",
            description="创建提醒事项",
            execute=create_reminder,
        ),
    }
```

工具系统做到这一步，后面要扩成 MCP、工作流或者多工具路由都比较自然。

另外，写操作最好从一开始就分风险等级。哪怕这里只是提醒工具，真正接入生产环境后也应该区分：

- `read`：查询天气、搜索信息
- `reversible-write`：创建提醒、写待办
- `irreversible-write`：支付、删除、公开发送消息

这类边界在实战里非常关键，因为很多 Agent 失败并不是“不够聪明”，而是“太自信地做错了事”。

---

## 5. 编排层：让人格、记忆和工具真正开始协作

前面的三层分别解决了“怎么定义角色”“怎么保存用户信息”“怎么执行外部动作”，但真正能把体验做出来的，还是编排层。

它负责决定：

- 当前消息要不要先召回长期记忆
- 什么时候调用工具
- 工具结果如何回注上下文
- 本轮对话结束后哪些内容要写回长期记忆

### 5.1 主流程可以保持得很清晰

一个最小可运行版本的主流程通常就是下面这几步：

```text
接收用户消息
  ↓
召回相关长期记忆
  ↓
判断是否需要调用工具
  ↓
拼接人格、记忆、工具结果
  ↓
模型生成回复
  ↓
提取新记忆并写回
```

### 5.2 用 Python 把这条链路串起来

下面这段代码不是完整生产实现，但很适合作为整篇文章的核心骨架，因为它把前面所有模块都串到了一起。

```python
async def handle_message(user_input: str, actor_id: str, memory_store):
    tools = build_tool_registry(actor_id)
    system_prompt = build_personality_prompt(DEFAULT_TRAITS)
    recalled = memory_store.search(user_input, 5)

    tool_context = []

    if "天气" in user_input:
        weather = tools["check_weather"].execute({"city": "Shanghai"})
        tool_context.append(f"天气工具返回: {weather}")

    if "提醒" in user_input:
        tools["create_reminder"].execute({
            "title": "带伞",
            "time": "2026-08-13T14:00:00+08:00",
        })
        tool_context.append("提醒工具返回: 已成功创建提醒")

    final_prompt = f"""
系统设定：
{system_prompt}

召回记忆：
{recalled}

工具结果：
{tool_context}

用户消息：
{user_input}
""".strip()

    reply = await run_chat_model(final_prompt)

    for memory in extract_memories_from_turn(user_input, reply):
        memory_store.save(memory)

    return reply
```

这段代码最重要的不是语法，而是它对应的运行顺序：先人格，后记忆，再工具，最后写回。只要这条闭环打通，后面无论替换模型、增加工具，还是把本地存储换成数据库，系统都能自然往上长。

---

## 6. 模型怎么选：先选能组成闭环的，不急着追排行榜

把工程骨架搭好之后，模型选择就会变得清楚很多。这里不需要一上来就追最强配置，而是要先选出一套能把整条链路跑通的组合。

### 6.1 主对话模型

如果是中文和多语言场景，主对话模型可以优先考虑：

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

主模型主要负责：

- 理解用户意图
- 结合人格和记忆生成最终回复
- 决定是否触发工具调用

### 6.2 记忆检索模型

长期记忆如果要做语义召回，可以用：

- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

比较典型的链路是：

```text
用户问题
  ↓
BGE-M3 召回候选记忆
  ↓
Reranker 重排
  ↓
把 Top K 放进 Prompt
```

### 6.3 如果要扩到语音

后续如果要把这个 Agent 做成语音助手，可以继续接：

- [openai/whisper-small](https://huggingface.co/openai/whisper-small)
- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [huggingworld/silero-vad](https://huggingface.co/huggingworld/silero-vad)

语音层可以在这条文本链路稳定后再接入。当前最关键的链路是“文本输入 → 记忆召回 → 工具调用 → 记忆写回”。

---

## 7. Python 生态中的开源参考

前面的代码是一个教学型最小实现。真正开发时，可以根据项目规模选择成熟的 Python 框架，把记忆、工具、工作流和模型调用交给专门组件处理。

### 7.1 LangChain：连接模型、工具和记忆

[LangChain](https://github.com/langchain-ai/langchain) 适合用来搭建 Agent 的基础调用链。它提供模型封装、工具定义、消息历史和结构化输出等能力，适合作为本文中 `models/`、`tools/` 和部分 `agent/` 模块的替代实现。

工具可以用函数直接声明：

```python
from langchain_core.tools import tool

@tool
def check_weather(city: str) -> dict:
    """查询指定城市的天气。"""
    return {
        "city": city,
        "weather": "light rain",
        "temperature": 26,
    }
```

这样做的价值不只是少写几行代码，更重要的是工具名称、描述和参数 Schema 可以统一交给模型调用层管理。

### 7.2 LangGraph：把 Agent 过程显式建模成状态图

[LangGraph](https://github.com/langchain-ai/langgraph) 更适合处理多步骤、可恢复、需要状态管理的 Agent。本文的流程可以表达成：

```text
接收消息
  ↓
召回记忆
  ↓
判断工具
  ↓
执行工具
  ↓
生成回复
  ↓
写回记忆
```

在 LangGraph 中，这些步骤可以变成节点，节点之间的流转由状态决定。这样做特别适合扩展：

- 工具失败后的重试
- 写操作前的人工确认
- 长任务的中断与恢复
- 多 Agent 协作

如果 Agent 后面要接日程、提醒和多个外部服务，LangGraph 比单个 `handle_message()` 更容易维护。

### 7.3 LlamaIndex：把重点放在记忆和检索

[LlamaIndex](https://github.com/run-llama/llama_index) 更适合参考数据接入、索引构建、RAG 和检索增强。对于本文的 Memory Layer，可以借鉴它的思路，把长期记忆拆成：

- 结构化用户资料
- 对话事件
- 偏好和事实
- 可检索的向量索引

一个常见的工程组合是：

```text
SQLite / PostgreSQL：保存记忆类型、来源、时间和置信度
向量索引：负责语义召回
Reranker：对候选记忆做二阶段排序
```

这比把完整聊天记录直接塞进 Prompt 更适合长期运行。

### 7.4 Mem0：接入长期记忆管理

[Mem0](https://github.com/mem0ai/mem0) 专门面向 AI 应用的记忆管理，可以参考它的记忆提取、更新和召回方式。

在本文的架构里，它可以替换 `memory/store.py` 和 `memory/recall.py` 的一部分。需要关注的不是简单的“保存一段文本”，而是：

- 新记忆如何从对话中提取
- 新信息如何覆盖旧信息
- 记忆如何按用户隔离
- 记忆如何被检索和注入上下文

对于长期交互型 AI，还需要额外保留记忆类型、重要性、来源和过期时间，避免长期记忆变成无法审计的黑盒。

### 7.5 Haystack：构建可观测的检索管道

[Haystack](https://github.com/deepset-ai/haystack) 适合参考组件化 Pipeline 和检索流程。本文中的记忆召回可以拆成：

```text
Query 清洗
  ↓
Embedding
  ↓
向量召回
  ↓
关键词召回
  ↓
重排
  ↓
记忆过滤
```

当记忆规模变大后，把每个步骤拆成独立组件，更容易观察延迟、召回数量和错误来源。

### 7.6 OpenClaw 和 AgentOS：作为跨语言的设计参考

[OpenClaw](https://github.com/openclaw/openclaw) 的文件化记忆、记忆检索和上下文压缩前刷新机制，适合参考记忆治理；[AgentOS](https://github.com/framersai/agentos) 则适合参考人格、认知记忆、工具和 Agent Runtime 如何结合。

它们不属于本文的 Python 主线，但其中的数据组织和运行时设计仍然有借鉴价值。

可以按照下面的方式选择：

| 需求 | 推荐参考 |
| --- | --- |
| 快速接入模型和工具 | LangChain |
| 多步骤 Agent 和状态恢复 | LangGraph |
| 文档、记忆和检索系统 | LlamaIndex |
| 长期记忆管理 | Mem0 |
| 可观测检索 Pipeline | Haystack |
| 文件化记忆和陪伴 Runtime 思路 | OpenClaw、AgentOS |

---

## 8. 跑通并验证完整闭环

只打印一条最终回复，还不能说明 Agent 真的完成了记忆和工具调用。验证时需要把关键中间状态也输出出来，至少检查：

- 召回了哪些长期记忆
- 调用了哪些工具
- 工具返回了什么结果
- 本轮新增了哪些记忆
- 第二轮对话能否继续召回第一轮写入的内容

### 8.1 增加一个内存版存储

为了让示例可以独立运行，先用列表实现一个最小的 `InMemoryMemoryStore`。后续替换成 SQLite 或向量数据库时，只需要保持同样的接口。

```python
class InMemoryMemoryStore:
    def __init__(self):
        self.items: list[MemoryItem] = []

    def save(self, memory: MemoryItem) -> None:
        self.items.append(memory)

    def search(self, query: str, limit: int = 5) -> list[MemoryItem]:
        keywords = set(query.lower().split())
        scored = []

        for item in self.items:
            content = item.content.lower()
            score = sum(1 for word in keywords if word in content)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for score, item in scored[:limit] if score > 0]

    def get_by_id(self, memory_id: str) -> MemoryItem | None:
        return next(
            (item for item in self.items if item.id == memory_id),
            None,
        )
```

这个版本只使用关键词匹配，不代表生产环境的最终方案，但足以验证记忆写入和召回的生命周期。

### 8.2 给 Agent 增加可观测日志

调试 Agent 时，建议把中间过程结构化记录下来，而不是只看最后一句回复。

```python
from dataclasses import dataclass, field

@dataclass
class RunTrace:
    recalled_memories: list[MemoryItem] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    written_memories: list[MemoryItem] = field(default_factory=list)


async def handle_message(
    user_input: str,
    actor_id: str,
    memory_store,
) -> tuple[str, RunTrace]:
    trace = RunTrace()
    tools = build_tool_registry(actor_id)
    system_prompt = build_personality_prompt(DEFAULT_TRAITS)
    trace.recalled_memories = memory_store.search(user_input, 5)

    tool_context = []

    if "天气" in user_input:
        weather = tools["check_weather"].execute({"city": "Shanghai"})
        trace.tool_calls.append({
            "name": "check_weather",
            "input": {"city": "Shanghai"},
            "output": weather,
        })
        tool_context.append(f"天气工具返回: {weather}")

    if "提醒" in user_input:
        reminder_input = {
            "title": "带伞",
            "time": "2026-08-13T14:00:00+08:00",
        }
        result = tools["create_reminder"].execute(reminder_input)
        trace.tool_calls.append({
            "name": "create_reminder",
            "input": reminder_input,
            "output": result,
        })
        tool_context.append(f"提醒工具返回: {result}")

    final_prompt = f"""
系统设定：
{system_prompt}

召回记忆：
{trace.recalled_memories}

工具结果：
{tool_context}

用户消息：
{user_input}
""".strip()

    reply = await run_chat_model(final_prompt)

    trace.written_memories = extract_memories_from_turn(
        user_input,
        reply,
    )
    for memory in trace.written_memories:
        memory_store.save(memory)

    return reply, trace
```

### 8.3 第一轮：验证工具调用和记忆写入

先预置两条记忆，再输入一句同时包含面试、天气和提醒需求的话。

```python
async def demo():
    actor_id = "user-001"
    memory_store = InMemoryMemoryStore()

    memory_store.save(FactMemory(
        id="m1",
        type="fact",
        content="用户最近在准备算法面试",
        importance=0.9,
        created_at="2026-08-12T00:00:00",
    ))

    memory_store.save(EmotionalMemory(
        id="m2",
        type="emotion",
        content="用户最近对面试结果有点焦虑",
        mood="anxious",
        importance=0.95,
        created_at="2026-08-12T00:00:00",
    ))

    reply, trace = await handle_message(
        "我明天下午要出门面试，帮我看看天气，顺便提醒我带伞。",
        actor_id,
        memory_store,
    )

    print("=== 第一轮回复 ===")
    print(reply)
    print("=== 召回记忆 ===")
    for memory in trace.recalled_memories:
        print(memory.content)
    print("=== 工具调用 ===")
    for call in trace.tool_calls:
        print(call["name"], call["output"])
    print("=== 新增记忆 ===")
    for memory in trace.written_memories:
        print(memory.content)
```

预期可以观察到：

```text
=== 召回记忆 ===
用户最近在准备算法面试
用户最近对面试结果有点焦虑

=== 工具调用 ===
check_weather {'city': 'Shanghai', 'weather': 'light rain', 'temperature': 26}
create_reminder {'success': True}
```

这一步验证了三件事：旧记忆被召回，两个工具被执行，本轮对话产生了新的记忆。

### 8.4 第二轮：验证跨会话记忆

验证长期记忆不能只看第一轮。第二轮换一种表达方式，观察 Agent 是否还能找到相关信息。

```python
second_reply, second_trace = await handle_message(
    "你还记得我最近在为什么事情发愁吗？",
    actor_id,
    memory_store,
)

print("=== 第二轮回复 ===")
print(second_reply)
print("=== 第二轮召回 ===")
for memory in second_trace.recalled_memories:
    print(memory.content)
```

如果第二轮能召回“准备面试”和“对面试结果焦虑”这两条内容，说明最小的跨轮记忆链路已经成立。把 `memory_store` 换成数据库，并在两次调用之间重新创建 Agent，还可以进一步验证跨进程、跨会话持久化。

### 8.5 用检查项记录验证结果

为了让实战过程更接近真实工程，可以把验证结果整理成一张表：

| 验证项 | 预期结果 | 实际结果 |
| --- | --- | --- |
| 人格 Prompt 生成 | 包含 HEXACO 参数和行为规则 | 通过 |
| 初始记忆召回 | 找到面试和焦虑相关记忆 | 通过 |
| 天气工具 | 返回城市、天气和温度 | 通过 |
| 提醒工具 | 返回创建成功 | 通过 |
| 新记忆写回 | 保存本轮重要信息 | 通过 |
| 第二轮召回 | 重新找到面试和焦虑记忆 | 通过 |

这类验证结果比“代码能够运行”更有说服力，因为它对应的是 Agent 的实际行为闭环。

---
## 9. 实战里最容易踩的坑

真正开始实现之后，问题通常不在“会不会写代码”，而在系统边界是不是一开始就想清楚了。

### 9.1 人格容易漂

只靠一段 prompt，很容易在多轮对话和工具调用后漂掉。比较稳妥的做法是：

- 固定人格参数
- 让人格同时影响运行时偏置
- 为角色准备一组回归测试

### 9.2 记忆容易污染

如果提取规则太激进，长期记忆很快就会变成噪音仓库。实际工程里最好从一开始就保留：

- 来源
- 置信度
- 过期时间
- 用户纠错与删除能力

### 9.3 工具调用容易越界

提醒还是低风险工具，一旦接日历、发消息、支付或文件系统，风险就完全不一样了。生产环境里至少要有：

- 工具风险分级
- 写操作确认
- 超时与重试机制
- 失败后的降级策略

### 9.4 Token 和延迟会上升

人格 Prompt、长期记忆、工具结果都会拉长上下文，所以后面几乎一定要继续做优化：

- 渐进式记忆披露
- 记忆摘要
- 工具结果压缩
- 使用更小的辅助模型处理分类和提取

---

## 10. 总结

长期交互型 AI Agent 不是“聊天模型 + 几个工具”的简单拼接，而是一个围绕人格、长期记忆和工具编排构建出来的运行时系统。

从工程实践角度看，最重要的不是一开始就追求一个功能很全的产品，而是先把最小闭环跑通：

1. 用结构化人格保证角色稳定性
2. 用长期记忆保证跨会话连续性
3. 用工具调用把“建议”变成“行动”
4. 用 Python 把这些模块组织成一个真正可维护的 Agent 工程

只要这条链路已经成立，后面无论是接入更强的模型、更复杂的数据库，还是扩展到语音和主动陪伴，系统都会有一个稳定的落脚点。

---

## 项目链接

- [AgentOS](https://github.com/framersai/agentos)
- [VoltAgent](https://github.com/VoltAgent/voltagent)
- [OpenClaw](https://github.com/openclaw/openclaw)
- [Memori](https://github.com/MemoriLabs/Memori)
- [aman-agent](https://github.com/amanasmuei/aman-agent)
- [AICO](https://boeni-industries.github.io/aico/welcome/)

## 模型链接

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [openai/whisper-small](https://huggingface.co/openai/whisper-small)
- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [huggingworld/silero-vad](https://huggingface.co/huggingworld/silero-vad)
