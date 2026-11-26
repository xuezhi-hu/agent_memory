# SparseMem | 三级记忆构建机制

**L0 Direct Memory · L1 Linked Memory · L2 Refined Memory**  
设计说明与 Codex 实现指南 · v0.1

本文是《SparseMem_Codex_Implementation.md》中记忆构建部分的细化说明，适用于现有长期记忆实验项目的增量开发。核心设计为最小必要记忆加工（Minimal Necessary Memory Construction）。下文的规则、分数、阈值与等级比例均为初始研究假设，不是实测结果。

## 1. 总体设计：渐进式判断，不做 LLM 三分类

每个 Memory Block 默认从 L0 开始。先使用非生成式方法检测是否存在未解决的上下文依赖；若存在，则尝试低成本补全；补全后仍具有较高残余依赖的块才进入 L2 候选池。只有预算允许且有足够原始证据的候选块才调用 LLM。

**核心算法：Detect → Resolve → Re-evaluate → Refine**

| 等级 | 判定条件 | 处理方式 | 构建期生成式 Token |
|---|---|---|---|
| L0 | 无明显未解决依赖 | 原文直接存储 | 0 |
| L1 | 有依赖，但可通过低成本方法充分补全 | 原文关联、时间标准化、元数据增强 | 0 |
| L2 | 补全后仍有显著复杂依赖，且预算允许 | LLM 局部叙事性事实精炼 | 有 |

如果预算不足、原始证据缺失或精炼失败，则保留原文与已找到的上下文，标记未解决依赖；不伪造事实，不将未执行精炼的候选记录为 L2。

### 1.1 基本处理单位

以语义分段后的 Memory Block 为处理单位，而非整个 Session 或单个句子。一个块可以包含一条或多条相关消息。分段利用相邻消息 Embedding 相似度、问答关系和指代信号；不要无条件拆开连续问答。

例如一个包含 20 条消息的 Session 被分为 B1（个人爱好，4 条）、B2（宠物信息，3 条）、B3（复杂工作计划变更，6 条）、B4（住址，2 条）、B5（旅行经历，5 条）。假设它们分别进入 L0、L1、L2、L0、L1，则只有 B3 可能需要 LLM 加工。这是示例而非真实分布。

## 2. L0：Direct Memory（直接记忆）

### 2.1 判定与存储

原始对话：`I moved to Beijing in August 2024 to work as a software engineer.`

若主体、事件、必要时间信息已经明确，且没有未解决的外部指代、语义或状态依赖，直接存储原文。第一人称 I 由说话者标记确定；并非所有陈述都必须包含时间或完整的主谓宾结构。

| 检测内容 | 初始实现 |
|---|---|
| 主体、事件与句法线索 | spaCy 依存句法分析 |
| 人名、地名、组织 | spaCy NER |
| 指代与简短回答 | 关键词、词性、前后问答结构 |
| 时间表达 | 正则、dateparser、Session 日期 |
| 语义依赖 | 本地 Embedding 相似度与问答规则 |
| 状态变更 | 关键词、历史实体匹配 |

候选触发词（仅为提示信号，不是自动判定条件）：

```python
PRONOUNS = ['he', 'she', 'it', 'they', 'this', 'that', 'those']
RELATIVE_TIME = ['yesterday', 'tomorrow', 'last week', 'next month']
UPDATE_WORDS = ['actually', 'instead', 'changed', 'no longer', 'used to']
DEPENDENT_WORDS = ['yes', 'no', 'also', 'because', 'exactly']
```

L0 只需要存储原文、说话者、证据指针、Embedding 与基本元数据，不进行生成式文本重写。

## 3. L1：Linked Memory（关联记忆）

L1 不生成新的自然语言事实，而是通过关联原始证据解决依赖。可以在索引文本中按原顺序拼接相关消息；原始档案单独保存，避免重复物理存储。

### 3.1 问答补全（Semantic）

- Assistant: What is your favorite programming language?
- User: Python.

识别简短回答及此前相关问题，关联两条消息，保留角色信息。检索文本可以是原始问答拼接，无须生成 “The user's favorite programming language is Python.”。

### 3.2 实体补全（Entity）

- User: I have a dog named Lucky.
- User: He is three years old.

通过前文实体、距离、类型兼容性等寻找候选。若候选足够明确，则关联两条原文；不必强行把 He 替换为 Lucky。多个候选无法区分时保留歧义。

### 3.3 时间补全（Temporal）

- Session 日期：2025-05-20。
- User: I bought a new laptop yesterday.
- 元数据：mention_time = 2025-05-20；event_time = 2025-05-19。

根据可靠的日期锚点解析相对时间。时间粒度不足或无法确定时保留原始表达，可将 event_time 设为 null 或记录时间区间，禁止猜测。

### 3.4 历史状态关联（Update）

- Session 1: I currently live in Beijing.
- Session 8: I have now moved to Shanghai.

根据主体、属性、时间顺序、更新词和语义相似度建立历史关联。保留北京与上海两条状态及原始证据，不覆盖或删除历史信息。相似度本身不能证明发生了状态更新，不能擅自确认不明确的变化。

## 4. L2：Refined Memory（精炼记忆）

只有在 L1 补全后仍然存在复杂依赖，且精炼能够帮助检索、证据充分并符合预算时，才调用 LLM。输入为目标块和必要前文，不默认输入整个 Session。

示例（原始对话）：

- User: I received offers from Google and Microsoft.
- Assistant: Which one are you leaning toward?
- User: The former, initially.
- Assistant: Has anything changed?
- User: Yes, the other one increased its offer by 20%, so I accepted it last Friday.

这里涉及 former / other one / it 的多重指代、选择变化及时间。可能的 L2 输出：

> The user received offers from Google and Microsoft. Initially, the user preferred Google, but ultimately accepted Microsoft's offer last Friday after Microsoft increased its offer by 20%.

若有可靠 Session 日期，可以进一步标准化 last Friday；否则保留原表达。保留原文及来源指针。此例说明复杂依赖可能有精炼价值，并不意味着每段类似对话都必须调用 LLM。

## 5. 核心算法：残余上下文依赖（RCD）

### 5.1 四维依赖向量

D(m) = [E, S, T, U]；其中 E 为实体依赖，S 为语义依赖，T 为时间依赖，U 为状态更新依赖。

| 分值 | 初始含义 |
|---|---|
| 0 | 未检测到未解决的依赖 |
| 0.5 | 可能存在未解决依赖或存在不确定性 |
| 1 | 存在明显未解决的依赖 |

初始平均评分 D = (E + S + T + U) / 4；低成本补全后重新评估四个维度，得到 R = (E' + S' + T' + U') / 4。第一版采用等权重，后续可消融其他权重。

| 例子 | 初始 [E,S,T,U] | 补全后 | 路由 |
|---|---|---|---|
| 完整陈述 | [0,0,0,0] | 不需要 | L0 |
| 简短问答 | [0.5,1,0,0] | [0,0,0,0] | L1 |
| 相对时间 | [0,0,1,0] | [0,0,0,0] | L1 |
| 复杂选择变更 | [1,1,0.5,1] | [1,0.5,0,1] | L2 候选 |

上述分值为人工演示，不是 NLP 实际输出。评估的对象是块内尚未解决的依赖，而不是关键词数量；一个指代词若已有明确的块内证据，不需要再标为高风险。

### 5.2 路由伪代码

```python
def construct_memory(block, history):
    initial = detect_dependency(block)
    if max(initial) == 0:
        return build_L0(block)

    linked = resolve_dependency(block, history)
    residual = detect_dependency(linked)
    score = sum(residual) / 4

    if score < 0.25 and max(residual) < 1:
        return build_L1(linked)

    return build_refinement_candidate(linked, score)
```

阈值 0.25 为初始配置。单项严重依赖不得被平均分掩盖；检测器需要读取补全后的证据及元数据，不能仅重复扫描原始文本。构建时 history 仅包含允许使用的历史对话，不读取测试问题、答案或证据标签。

### 5.3 候选分配和预算

候选池按 Priority = R(m) / estimated_token_cost(m) 从高到低分配预算。估算成本应计入系统 Prompt、目标块、上下文、输出预留。预算耗尽的候选依然保留原文与已关联证据，并标记 deferred；实际消耗和失败重试必须统一记录。此优先级是启发式策略，不保证得到理论最优分配。

## 6. 统一存储与检索

| 字段 | L0 | L1 | L2 |
|---|---|---|---|
| 原始文本 / 证据指针 | 保留 | 保留 | 保留 |
| 语义向量、实体、时间 | 有 | 有 | 有 |
| 关联上下文 | 无需 | 有 | 按需 |
| 生成式叙事事实 | 无 | 无 | 有 |

三个等级使用同一个记忆库，不需要三个数据库。保留原文检索入口，原文与精炼文本均可通过相同的混合检索机制检索，按源证据去重。

实现建议（补充字段，可适配已有结构）：

```json
{
  "memory_id": "M001", "session_id": "S005",
  "level": "L1", "source_ids": [5, 6, 7],
  "linked_source_ids": [3, 4],
  "initial_dependency": [0.5, 1, 0, 0],
  "residual_dependency": [0, 0, 0, 0],
  "refinement_status": "not_required",
  "refined_text": null, "generation_tokens": 0
}
```

## 7. Codex 实现清单

| 函数 | 责任 |
|---|---|
| segment_session() | 动态语义分段 |
| detect_dependency() | 四维初始或残余依赖检测 |
| resolve_entity() | 实体关联 |
| resolve_semantic() | 问答、上下文拼接 |
| resolve_time() | 时间标准化 |
| resolve_update() | 历史状态关联 |
| compute_residual() | 计算补全后的依赖评分 |
| allocate_budget() | 预算约束下的候选选择 |
| refine_memory() | LLM 局部叙事事实提取 |

### 7.1 开发与验收要求（补充）

优先复用现有仓库模块，保留原有方法和评测接口；将路由阈值、预算、分段长度配置化。先用上文案例实现离线单元测试，再测试歧义、多实体、预算不足、时间锚点缺失与 LLM 失败等情况。不要使用生成式模型完成 L0/L1 分类，先使用 mock 验证，仅批准后进行小规模真实 API 评测。默认不直接运行完整 500 题实验。

统计各等级实际块数、原始文本 Token 覆盖率、生成式输入/输出 Token、调用次数、构建耗时、检索召回率和问答准确率。对比 Raw RAG、全量提取、同预算随机选择与 SparseMem。构建阶段不得利用测试问题或答案；各组使用相同回答模型与检索预算。

### 7.2 预期效果和限制

假设 L0 占 50%、L1 占 35%、L2 占 15%，则 85% 的记忆块不需要生成式加工。但这不意味着节省 85% 的 Token，因为块长度、Prompt 开销和推理成本不同。L0/L1 仍有本地 NLP 和 Embedding 计算开销。

真正的研究贡献是先检测依赖、低成本关联补全、重新评估残余依赖、最后选择性精炼的渐进式构建机制；其有效性与原创性仍需实验及文献验证。
