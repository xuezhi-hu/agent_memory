# 参考源码与适配边界

## LightMem 主题分段

参考 `../baselines/naive-rag-lightmem/` 提交 `09b57f7` 的
`src/lightmem/factory/topic_segmenter/llmlingua_2.py` 和
`src/lightmem/factory/memory_buffer/sensory_memory.py`。MIT 许可证原文见
`licenses/LightMem-MIT.txt`。本版本使用注意力候选边界和语义相似度调整，
适配为逐条人物发言；可以选择 LLMLingua-2 隐状态或独立 MiniLM 向量。
两者都未使用其原文预压缩，独立 MiniLM 版本与原项目的消息配对方式仍不同。

本实现为 DPC 自有工作流；参考仓库保留原状。A/B 是机制对照，不代表官方 SimpleMem 或 TSM 的完整复现。

## SimpleMem

本地参考：`../SimpleMem/`。

核对提交：`db80b6a7c591e0ea730a058e9f5fc4eb06572299`。

许可：MIT；Copyright 2025 AIMING Lab；原文附于 `licenses/SimpleMem-MIT.txt`。

- `simplemem/core/memory_builder.py`：借鉴窗口处理、紧凑独立记忆、独立命题提示思路；DPC 的共享事实 → 候选卡 → Guard 是本项目独立分阶段实现。
- `simplemem/core/models/memory_entry.py`：借鉴独立陈述及来源元数据表示；DPC 自行定义正文＋必要限定＋来源引用的卡片。
- `simplemem/core/hybrid_retriever.py`：借鉴语义/关键词多视图检索思路。`_merge_and_deduplicate_entries` 的稳定 ID 去重模式经适配用于 `dpc_mem/retrieval.py::deduplicate_entries`；保留许可。
- 本项目使用 NumPy cosine、BM25 与两层 RRF；未接入参考仓库的自由规划、反思、额外图数据库或完整依赖栈。

仓库可能含后续扩展；不把 EvolveMem/OmniSimpleMem 的模块一概归为原始 SimpleMem 论文贡献。

## TSM 与本项目既有适配

参考仓库：`../../DashMem/references/TSM/`。

核对提交：`0da3633ba8e27e9f8122872ff775587116948f5e`；MIT 原许可附于 `licenses/TSM-MIT.txt`。

- `dpc_mem/data.py` 适配自本项目 `DashMem/dashmem/datasets.py` 的数据分离方式。
- `dpc_mem/evaluation.py` 适配自 `DashMem/dashmem/evaluation.py` 的题型路由与评分解析。
- `dpc_mem/rubrics/*.txt` 从 `DashMem/docs/rubrics/` 逐字复制，沿用已核对的 TSM 评分模板；没有增加额外 system 评分标准。来源与哈希见 `rubrics/provenance.json`（文件名以目录实际清单为准）。
- 效率统计参考 TSM 构建/更新 token 和检索 P50/P95 口径，但实现为独立日志器，区分实际账单、共享步骤归因、失败和未知用量。
- 本方法未搬用 TSM 的 TKG、月度聚合、GMM、人物手册或时效图更新逻辑。

## LeanMem

本地论文：`../LeanMem_论文_arxiv_2608.03463v1.pdf`。

源码状态见 `../LeanMem_源码核查说明.md`：本次没有取得并适配经确认的官方实现。仅借鉴“按内容保真要求选择压缩、保留细节证据”的设计思路；不得写成已复用 LeanMem 源码。

## DPC 自写核心

- 针对已经生成的候选综合定位实际丢失的区别。
- 比对原文、事实、候选，区分抽取错误与压缩错误。
- ACCEPT / PATCH / SPLIT，一窗口一次批量检查；引用错误或不确定时回退。
- 必要限定作为不可拆分的索引与返回单元。
- 共享产物的 A/B/C/D 对照、M/MR 诊断、独立人工保留率统计。

这些是实现归属说明，不以“代码自写”替代论文新颖性论证。
