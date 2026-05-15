# 大模型学习路线图与资源清单（2026 版）

> 版本日期：2026-05-15  
> 视角：AI 大模型专家 / LLM 应用架构师 / 训练与推理工程师  
> 目标：建立从底层原理、模型训练、后训练、RAG、Agent、多模态、推理部署、评测安全到工程落地的完整知识体系。  
> 适合对象：已有 Python 基础，希望系统掌握大模型技术的人；也适合算法、后端、数据、产品、架构师转向大模型方向。

---

## 0. 先给结论：应该按什么顺序学？

大模型知识很多，但学习顺序不能按“热搜榜”来。推荐顺序是：

```text
Python / PyTorch / 数学基础
        ↓
Transformer / 语言模型 / Tokenizer / 预训练基本逻辑
        ↓
Prompt & Context Engineering / API / 结构化输出 / 工具调用
        ↓
RAG / 向量检索 / 文档解析 / 评测
        ↓
Agent / MCP / A2A / 工作流编排 / 记忆 / 多工具协作
        ↓
LoRA / QLoRA / SFT / DPO / RLHF / GRPO 等后训练方法
        ↓
推理服务 / 量化 / KV Cache / vLLM / TGI / TensorRT-LLM / llama.cpp
        ↓
多模态 / 长上下文 / 推理模型 / 代码智能体 / 安全治理 / LLMOps
        ↓
Capstone 项目：做出能上线、能评测、能监控、能降本的大模型系统
```

**企业落地优先级建议：**

1. **先学会用好模型**：Prompt、结构化输出、函数调用、RAG、评测、成本控制。
2. **再学会改造模型**：SFT、LoRA、DPO、蒸馏、数据构造。
3. **最后再深入训练大模型**：预训练、分布式训练、MoE、Scaling Law、数据治理、GPU 集群调度。

大多数公司并不需要从零训练基础模型，但几乎都需要会做 **RAG、Agent、评测、安全、部署、微调和成本优化**。

---

## 1. 大模型知识全景图

### 1.1 五层能力模型

| 层级 | 你需要掌握什么 | 代表能力 | 是否必须 |
|---|---|---|---|
| L1 基础能力 | Python、PyTorch、数学、Linux、Docker、Git、API | 能读懂模型代码，能写训练和推理脚本 | 必须 |
| L2 模型原理 | Transformer、Attention、Tokenizer、位置编码、预训练目标、Scaling Law | 能解释大模型为什么能生成、为什么会幻觉、为什么长上下文难 | 必须 |
| L3 应用工程 | Prompt、RAG、函数调用、结构化输出、Agent、MCP、向量库、文档解析 | 能做企业知识库、AI 助手、数据分析助手、工作流自动化 | 必须 |
| L4 训练与优化 | SFT、LoRA、QLoRA、DPO、RLHF、GRPO、蒸馏、量化、推理加速 | 能定制模型、降成本、提升特定任务表现 | 进阶必须 |
| L5 生产治理 | Eval、红队、安全、隐私、监控、LLMOps、模型路由、成本与质量 A/B | 能把 demo 变成稳定产品 | 必须 |

---

## 2. 当前热门技术雷达

### 2.1 P0：所有大模型从业者都应该掌握

| 技术 | 你要掌握的关键词 | 学到什么程度 |
|---|---|---|
| Transformer | Self-Attention、MHA、GQA、MQA、MLA、FFN、LayerNorm、Residual、RoPE | 能手写一个简化版 Transformer block |
| Tokenizer | BPE、SentencePiece、特殊 token、chat template、上下文长度 | 能训练一个小 tokenizer 并理解 token 成本 |
| Prompt Engineering | system prompt、few-shot、分解任务、约束输出、反例、rubric | 能稳定控制输出格式和任务流程 |
| Context Engineering | 上下文选择、压缩、记忆、长期状态、缓存、冲突处理 | 能把有限 context 用在最重要信息上 |
| Structured Output | JSON Schema、函数调用、Pydantic/Zod、校验、重试 | 能把模型输出接进后端系统 |
| RAG | embedding、chunking、hybrid search、reranker、query rewrite、citation、grounding | 能做可评测、可溯源的知识库 |
| Agent | ReAct、plan-act-reflect、tool calling、memory、handoff、human-in-the-loop | 能做多步骤任务自动化 |
| Eval | golden set、LLM-as-judge、RAGAS、offline eval、online eval、A/B、人工评审 | 能证明系统变好了，而不是“感觉变好了” |
| LLMOps | tracing、prompt version、model routing、fallback、rate limit、cost dashboard | 能上线、监控、回滚、降本 |
| 安全 | prompt injection、data leakage、jailbreak、tool abuse、excessive agency | 能做基本防护与红队测试 |

### 2.2 P1：应用与工程落地热门方向

| 技术 | 为什么热门 | 建议掌握 |
|---|---|---|
| Agentic RAG | 传统 RAG 对复杂问题不够，Agent 可以决定何时检索、检索哪里、是否二次检索 | LangGraph / LlamaIndex / Haystack 任取一个深入 |
| MCP | 用统一协议把模型应用连接到文件、数据库、搜索、业务系统、工具 | 会写 MCP server/client，理解权限与沙箱 |
| A2A | 多 Agent 跨系统通信与协作的开放协议 | 理解 agent card、任务发现、消息协商 |
| AI Coding Agent | 代码理解、补丁生成、测试执行、PR 自动化 | 会做 repo indexing、工具调用、单元测试闭环 |
| 多模态 RAG | 图片、表格、PDF、视频、语音进入知识库 | 掌握 OCR、layout parser、VLM、caption、multimodal embedding |
| GraphRAG | 把实体关系、知识图谱与向量检索结合 | 能处理复杂组织知识、关系型知识和长链推理 |
| SQL/BI Agent | 自然语言查数、生成 SQL、解释指标 | 必须做权限、SQL 校验、只读沙箱、审计 |
| LLM Gateway | 多模型统一接口、路由、限流、fallback、成本跟踪 | LiteLLM / 自建 gateway |
| Observability | 追踪每次模型调用、工具调用、检索结果、用户反馈 | LangSmith / OpenTelemetry / 自建日志 |
| Prompt/Program Optimization | 不靠手工调 prompt，而是用数据和指标优化 | DSPy、promptfoo、自动化评测 |

### 2.3 P2：算法、训练和研究前沿

| 技术 | 需要关注的点 |
|---|---|
| Reasoning Models | test-time compute、reasoning tokens、self-verification、RLVR、GRPO、long-horizon planning |
| MoE | router、expert parallelism、load balancing、active parameters、MoE fine-tuning |
| 长上下文 | RoPE 扩展、KV cache 管理、attention 优化、context compression、长文档 eval |
| 后训练算法 | SFT、RLHF、PPO、DPO、IPO、KTO、ORPO、GRPO、RLAIF、constitutional AI |
| 合成数据 | self-instruct、data distillation、teacher-student、rejection sampling、quality filtering |
| 推理加速 | FlashAttention、PagedAttention、continuous batching、speculative decoding、prefix cache、quantization |
| 小模型与端侧模型 | 蒸馏、量化、GGUF、llama.cpp、ONNX、WebGPU、移动端推理 |
| 多模态模型 | VLM、OCR-free doc understanding、speech-to-speech、video reasoning、VLA/robotics |
| 模型可解释与安全 | mechanistic interpretability、red teaming、jailbreak eval、model behavior auditing |
| 训练基础设施 | FSDP、DeepSpeed、Megatron、Ray、Kubernetes、GPU 拓扑、checkpoint、数据吞吐 |

---

## 3. 必备基础知识

### 3.1 编程与工程基础

必须掌握：

- Python：函数、类、装饰器、生成器、异步、类型注解、异常处理。
- PyTorch：Tensor、autograd、Dataset/DataLoader、nn.Module、optimizer、mixed precision。
- Linux：shell、进程、端口、日志、文件权限、GPU 状态查看。
- Git：分支、PR、rebase、tag、submodule。
- Docker：镜像、容器、volume、GPU runtime、docker compose。
- API：HTTP、REST、JSON Schema、鉴权、超时、重试、流式输出。
- 数据工程：Parquet、JSONL、CSV、ETL、数据清洗、去重、版本管理。

推荐实践：

1. 写一个 PyTorch MLP/CNN/RNN 小模型。
2. 写一个 FastAPI 服务，提供 `/chat`、`/health`、`/metrics`。
3. 用 Docker 部署一次本地服务。
4. 用 JSONL 管理训练/评测数据。

### 3.2 数学基础

不用把数学学成研究生考试，但要能读懂核心公式：

| 主题 | 需要掌握 |
|---|---|
| 线性代数 | 向量、矩阵乘法、特征值、低秩分解、范数、相似度 |
| 概率统计 | 条件概率、交叉熵、KL 散度、最大似然、采样 |
| 优化 | SGD、Adam、学习率、warmup、weight decay、梯度裁剪 |
| 信息论 | entropy、perplexity、bits-per-byte、compression view |
| 数值计算 | FP32/FP16/BF16/FP8、溢出、混合精度、量化误差 |

---

## 4. 模型原理：从语言模型到 Transformer

### 4.1 语言模型基本概念

你需要能解释：

- 什么是自回归语言模型：预测下一个 token。
- 为什么训练目标是 next-token prediction。
- perplexity 代表什么。
- temperature、top-k、top-p、beam search、repetition penalty 如何影响生成。
- 为什么模型可能“幻觉”：训练目标不是事实数据库，而是条件概率建模。
- 为什么上下文长度有限：Attention 和 KV cache 成本随上下文增长。

### 4.2 Tokenizer

重点：

- BPE、WordPiece、SentencePiece 的区别。
- token 与字符、单词不一一对应。
- 中文、代码、表格、特殊符号的 token 成本。
- chat template：system/user/assistant/tool role 如何被拼成 token。
- 特殊 token：BOS、EOS、PAD、tool call token、image token。

练习：

- 用 Hugging Face Tokenizers 训练一个 BPE tokenizer。
- 比较中文、英文、代码、Markdown 的 token 数。
- 写一个函数估算 prompt 成本和输出成本。

### 4.3 Transformer

必须理解：

- Embedding。
- Self-Attention。
- Multi-Head Attention。
- Feed Forward Network。
- Residual Connection。
- LayerNorm / RMSNorm。
- Positional Encoding / RoPE / ALiBi。
- Causal Mask。
- Pre-norm 与 post-norm。
- Decoder-only、Encoder-only、Encoder-Decoder 的区别。

建议手写：

- 一个最小 GPT block。
- 一个小型字符级语言模型。
- 一个简单 generate 函数，支持 temperature/top-p。

### 4.4 架构演进

| 技术 | 作用 |
|---|---|
| MQA / GQA | 降低 KV cache 成本，提高推理吞吐 |
| MLA | 用低秩 latent 表示压缩 KV 相关信息，提升推理效率 |
| MoE | 激活少量专家，扩大总参数但控制每 token 计算量 |
| RoPE 扩展 | 支持更长上下文 |
| Sliding Window Attention | 降低长序列注意力成本 |
| FlashAttention | 优化 attention 的显存访问和计算效率 |
| Multi-token Prediction | 一次预测多个未来 token，可能提升训练/推理效率 |
| Speculative Decoding | 小模型草稿，大模型验证，提升解码速度 |
| KV Cache Quantization | 降低长上下文推理显存 |

---

## 5. 数据与预训练

### 5.1 预训练数据

要理解：

- 数据来源：网页、书籍、代码、论文、问答、数学、合成数据、多模态数据。
- 数据清洗：去重、语言识别、质量过滤、毒性过滤、PII 过滤、版权风险。
- 数据配比：通用能力、代码能力、数学能力、多语言能力、领域能力之间的 trade-off。
- 数据污染：benchmark contamination、train-test leakage。
- 数据版本：训练集、SFT 数据、偏好数据、评测集要严格隔离。

### 5.2 Scaling Law

核心思想：

- 模型规模、数据规模、计算量之间存在经验规律。
- 给定算力，参数越大不一定越好；数据 token 量也很关键。
- 推理成本正在成为模型选择的重要约束。

你应该能回答：

1. 为什么小模型加高质量数据和蒸馏有时能接近大模型？
2. 为什么训练一个基础模型比微调贵几个数量级？
3. 为什么长上下文模型不一定比 RAG 更适合所有场景？

### 5.3 从零预训练是否值得学？

值得理解，不一定要大规模实践。

建议实践：

- 用 TinyStories 或小型中文语料训练一个小 GPT。
- 重点观察 loss、perplexity、过拟合、采样效果。
- 不追求效果，追求理解训练流程。

---

## 6. 后训练、对齐与微调

### 6.1 后训练流程

典型流程：

```text
Base Model
  ↓
Instruction Tuning / SFT
  ↓
Preference Data
  ↓
Reward Model 或 Direct Preference Optimization
  ↓
RLHF / DPO / GRPO / RLAIF
  ↓
Safety Tuning
  ↓
Evaluation & Red Teaming
```

### 6.2 SFT：监督微调

你需要掌握：

- 指令数据格式：单轮、多轮、工具调用、多模态。
- chat template 必须与推理时一致。
- loss mask：只对 assistant 输出算 loss。
- 数据质量通常比数量更重要。
- SFT 适合教格式、风格、领域知识、工具调用模式，但不适合“塞进大量事实知识”。

练习：

- 用 LoRA 微调一个 1B-7B 开源模型，让它学会固定 JSON 输出。
- 构造 200-1000 条高质量领域问答，比较微调前后表现。
- 做 train/eval split，避免只看训练集效果。

### 6.3 LoRA / QLoRA / PEFT

重点：

- LoRA：冻结原模型，只训练低秩 adapter。
- QLoRA：把基座模型量化到 4-bit，再训练 adapter。
- rank、alpha、target modules、dropout、learning rate 对效果的影响。
- adapter merge 与不 merge 的部署差异。
- PEFT 的收益：降低显存、训练速度更快、便于多任务适配。

常见错误：

- 数据太少却训练太久，导致过拟合。
- 学习率过大，输出格式崩坏。
- 微调用来记知识，而不是配合 RAG。
- 推理时 chat template 与训练不一致。
- 只看主观 demo，不做 eval。

### 6.4 DPO / RLHF / GRPO

| 方法 | 核心思想 | 适合场景 |
|---|---|---|
| RLHF | 用人类偏好训练奖励模型，再用 RL 优化策略 | 对齐复杂行为，但工程成本高 |
| DPO | 不显式训练奖励模型，直接用偏好对优化 | 成本更低，常用于偏好对齐 |
| RLAIF | 用 AI 反馈替代部分人工反馈 | 节省标注成本 |
| GRPO | 近年在 reasoning/RLVR 中常见，强调按组比较与奖励优化 | 数学、代码、可验证任务 |
| Constitutional AI | 用原则/宪法式规则引导模型行为 | 安全、价值观、拒答策略 |

### 6.5 什么时候该微调，什么时候该 RAG？

| 问题 | 优先方案 |
|---|---|
| 模型不知道企业私有文档 | RAG |
| 输出格式总是不稳定 | 结构化输出 + SFT |
| 需要固定语气/品牌风格 | SFT |
| 需要掌握大量实时知识 | RAG / tool |
| 需要学会调用内部工具 | tool calling 数据 SFT + Agent eval |
| 需要提升数学/代码推理 | 先换更强 reasoning model；再考虑 RL/蒸馏 |
| 需要降低成本 | 蒸馏 / 小模型微调 / 路由 / 缓存 |

---

## 7. Prompt 与 Context Engineering

### 7.1 Prompt Engineering

必须掌握：

- 指令清晰：任务、约束、输入、输出、评价标准。
- few-shot 示例。
- 反例与边界条件。
- 分步任务拆解。
- 输出格式约束。
- 引用与证据要求。
- 失败时重试策略。

推荐模板：

```text
角色：你是……
目标：完成……
输入：……
约束：
1. 不要……
2. 必须……
输出格式：
{
  "field": "...",
  "confidence": 0.0,
  "evidence": [...]
}
评价标准：
- 正确性
- 完整性
- 可追溯性
- 简洁性
```

### 7.2 Context Engineering

现在比单纯 prompt 更重要。

你要掌握：

- 什么信息该进入上下文？
- 什么信息该被压缩？
- 多轮对话如何保留状态？
- RAG 检索内容如何排序？
- 多工具结果冲突时如何处理？
- 上下文过长时如何裁剪？
- 是否使用长期记忆？
- 记忆如何授权、更新和删除？

实践：

- 做一个会议纪要助手：输入多轮聊天，输出可追踪任务清单。
- 做一个上下文压缩器：把长对话压缩成 state object。
- 做一个 memory policy：什么信息允许记忆，什么必须遗忘。

---

## 8. RAG：企业落地核心技术

### 8.1 标准 RAG 流程

```text
文档收集
  ↓
清洗 / OCR / 表格解析 / 分段
  ↓
chunking
  ↓
embedding
  ↓
向量库 / 倒排索引 / 混合检索
  ↓
rerank
  ↓
prompt 拼接
  ↓
生成答案
  ↓
引用 / 置信度 / 评测 / 用户反馈
```

### 8.2 必须掌握的 RAG 技术

| 模块 | 技术点 |
|---|---|
| 文档解析 | PDF、HTML、Markdown、Office、表格、图片、OCR、layout-aware parsing |
| Chunking | fixed-size、semantic chunking、recursive split、section-aware split |
| Embedding | dense embedding、sparse embedding、multilingual embedding、domain embedding |
| 检索 | vector search、BM25、hybrid search、metadata filter |
| Rerank | cross-encoder reranker、LLM rerank、reciprocal rank fusion |
| Query Rewrite | query expansion、HyDE、多轮上下文补全 |
| 生成 | grounding、citation、answer synthesis、abstention |
| 评测 | context precision、context recall、faithfulness、answer relevance |
| 安全 | 权限过滤、租户隔离、引用检查、prompt injection 防护 |

### 8.3 高级 RAG

- **Agentic RAG**：模型决定是否检索、检索哪个库、是否继续检索。
- **GraphRAG**：实体关系图 + 向量检索，适合复杂关系和全局摘要。
- **Multi-hop RAG**：需要多次检索才能回答。
- **Multimodal RAG**：图片、表格、PDF layout、视频片段参与检索。
- **SQL RAG / Text-to-SQL**：自然语言查询数据库。
- **Tool-augmented RAG**：检索之外还能调用计算器、搜索、API。
- **Long-context RAG**：长上下文模型 + 检索压缩结合。
- **Self-correcting RAG**：生成后检查引用和事实一致性。

### 8.4 向量数据库和检索组件

选型原则：

| 类型 | 代表 | 适合场景 |
|---|---|---|
| 本地原型 | FAISS、Chroma | 快速 demo、本地实验 |
| 开源生产 | Milvus、Qdrant、Weaviate、pgvector、Elasticsearch/OpenSearch | 企业可控部署 |
| 云服务 | Pinecone、Vertex AI Vector Search、Azure AI Search 等 | 快速上线，少运维 |
| 混合检索 | Elasticsearch/OpenSearch + dense vector | 文档检索、关键词强约束 |

建议：先用 **FAISS/Chroma 做原型**，再根据生产要求选择 Milvus/Qdrant/pgvector/Elasticsearch。

---

## 9. Agent、工具调用、MCP 与 A2A

### 9.1 Agent 的本质

Agent 不是“会聊天的机器人”，而是：

```text
LLM + 状态 + 工具 + 计划 + 观察 + 记忆 + 评测 + 权限控制
```

常见模式：

- ReAct：Reason + Act，边想边调用工具。
- Plan-and-Execute：先规划，再执行。
- Reflection：执行后自我检查。
- Critic / Verifier：另一个模型或规则系统做审核。
- Handoff：把任务交给专门 Agent。
- Human-in-the-loop：高风险步骤需要人确认。
- Workflow Agent：用图或状态机约束 Agent 行为。

### 9.2 工具调用与结构化输出

必须掌握：

- Function calling / tool calling。
- JSON Schema。
- 参数校验。
- 工具结果回传。
- 工具失败重试。
- 超时和幂等。
- 权限和审计。
- 工具调用不等于直接执行，真正执行必须在应用层控制。

### 9.3 MCP

MCP 可以理解为大模型应用连接外部系统的标准接口。你要掌握：

- MCP server：暴露文件、数据库、搜索、业务 API、工具。
- MCP client：模型应用侧连接 MCP server。
- 资源、工具、prompt 的差异。
- 权限、沙箱、日志、最小授权。
- prompt injection 与 tool poisoning 风险。

练习：

1. 写一个本地文件搜索 MCP server。
2. 写一个数据库只读查询 MCP server。
3. 给工具加 allowlist、日志和参数校验。
4. 做一次 prompt injection 测试。

### 9.4 A2A

A2A 更偏向多 Agent 之间的通信与协作。你需要理解：

- agent card：一个 Agent 如何描述自己的能力。
- discovery：如何发现可用 Agent。
- task lifecycle：任务如何创建、更新、完成。
- message / artifact：Agent 之间如何传递内容。
- 安全与权限：不是所有 Agent 都该互相调用。

### 9.5 Agent 工程陷阱

| 陷阱 | 对策 |
|---|---|
| Agent 无限循环 | max steps、预算、超时、状态机 |
| 工具乱调用 | 工具选择 eval、权限控制、必要时规则路由 |
| 幻觉式调用 | JSON Schema、参数校验、工具结果 grounding |
| 越权操作 | human approval、RBAC、只读默认 |
| 成本爆炸 | 模型路由、缓存、步骤限制、批处理 |
| 不可复现 | tracing、prompt version、seed、完整日志 |
| 只会 demo | golden task set、回归测试、线上监控 |

---

## 10. 多模态大模型

### 10.1 必须掌握

| 方向 | 重点 |
|---|---|
| VLM | 图像理解、图文问答、OCR、图表理解 |
| 文档智能 | PDF layout、表格、公式、图片、脚注、引用 |
| 语音 | ASR、TTS、speech-to-speech、实时语音 Agent |
| 视频 | frame sampling、video caption、temporal reasoning |
| 多模态 RAG | 图片/表格/文本共同检索，跨模态引用 |
| 多模态安全 | 图像 prompt injection、OCR 注入、deepfake、隐私泄露 |

### 10.2 推荐实践

1. 做一个 PDF 合同问答系统，能处理文字、表格和图片。
2. 做一个图表解释助手：上传图表，输出结论和不确定性。
3. 做一个多模态客服：图片 + 文本描述，判断问题并调用工具。
4. 做一个视频摘要系统：抽帧 + ASR + 章节总结。

---

## 11. 推理、部署与成本优化

### 11.1 推理服务核心概念

| 概念 | 说明 |
|---|---|
| Prefill | 处理输入 prompt 的阶段 |
| Decode | 逐 token 生成输出的阶段 |
| KV Cache | 保存历史 key/value，避免重复计算 |
| Batch / Continuous Batching | 合并请求提升 GPU 利用率 |
| Streaming | 边生成边返回 |
| Tensor Parallel | 模型切到多卡 |
| Pipeline Parallel | 层切到多卡 |
| Expert Parallel | MoE 专家并行 |
| Quantization | int8/int4/fp8 等降低显存和成本 |
| Speculative Decoding | 小模型生成草稿，大模型验证 |
| Prefix Cache | 相同前缀复用计算 |
| LoRA Serving | 一个基座模型挂多个 adapter |

### 11.2 常见推理框架

| 工具 | 适合 |
|---|---|
| vLLM | 高吞吐服务、OpenAI-compatible API、PagedAttention、continuous batching |
| Hugging Face TGI | Hugging Face 生态部署、生产推理服务 |
| TensorRT-LLM | NVIDIA GPU 上高性能推理优化 |
| llama.cpp | 本地/端侧/CPU/GPU 混合推理，GGUF 生态 |
| Ollama | 本地模型体验和轻量应用 |
| SGLang | 结构化生成、复杂 serving 场景 |
| LMDeploy | 开源模型部署、量化、推理服务 |
| Triton Inference Server | 多模型服务、企业推理部署 |
| ONNX Runtime / OpenVINO | CPU/边缘推理优化 |

### 11.3 成本优化路线

优先级：

1. 减少无效请求：缓存、去重、用户输入校验。
2. 减少 token：prompt 压缩、上下文裁剪、检索更准。
3. 模型路由：简单任务用小模型，复杂任务用大模型。
4. 批处理和流式返回。
5. 量化：int8/int4/GGUF。
6. 使用 vLLM/TGI/TensorRT-LLM 等高吞吐服务。
7. 蒸馏小模型。
8. 只在必要时微调。
9. 严格 eval，避免为无效提升买单。

### 11.4 生产部署清单

上线前至少具备：

- API 超时、重试、熔断。
- prompt 与模型版本记录。
- 请求日志脱敏。
- 用户权限过滤。
- 工具调用审计。
- token 成本统计。
- tracing。
- eval regression。
- 安全规则与人工复核。
- fallback 模型。
- 灰度发布。
- 监控报警。

---

## 12. 评测、安全与治理

### 12.1 为什么评测是核心能力？

大模型系统最常见的问题不是“不能做”，而是：

- 今天能做，明天换 prompt 后不能做。
- demo 很好，真实用户问题很差。
- 主观感觉提升，但没有指标。
- RAG 找到了错文档，模型答得很自信。
- Agent 调用了危险工具。
- 成本上升，但质量没有提升。

### 12.2 评测层次

| 层次 | 评测内容 |
|---|---|
| 模型级 | 通用能力、数学、代码、多语言、安全 |
| Prompt 级 | 输出格式、准确性、稳定性 |
| RAG 检索级 | recall、precision、reranker 效果 |
| RAG 生成级 | faithfulness、citation correctness、answer relevance |
| Agent 级 | task success、tool accuracy、step count、cost、failure mode |
| 产品级 | 用户满意度、解决率、人工介入率、时延、成本 |
| 安全级 | prompt injection、jailbreak、数据泄露、越权调用 |

### 12.3 常用评测工具

- Ragas：RAG/Agentic workflow 评测。
- DeepEval：LLM 应用评测。
- promptfoo：prompt 和模型回归测试。
- LangSmith：LangChain/LangGraph tracing 与 eval。
- OpenAI Evals：模型与应用评测思路参考。
- HELM：通用模型评测框架。
- MLCommons AILuminate：AI 安全基准。
- 自建 golden set：生产中最重要。

### 12.4 安全风险

参考 OWASP LLM Top 10，重点关注：

1. Prompt Injection。
2. Sensitive Information Disclosure。
3. Supply Chain Vulnerabilities。
4. Data and Model Poisoning。
5. Improper Output Handling。
6. Excessive Agency。
7. System Prompt Leakage。
8. Vector and Embedding Weaknesses。
9. Misinformation。
10. Unbounded Consumption。

### 12.5 Agent 安全原则

- 默认只读。
- 高风险动作必须人工确认。
- 工具最小权限。
- 参数白名单。
- 结果校验。
- 业务系统二次鉴权。
- 日志审计。
- 可回滚。
- 明确预算和 step limit。
- 不把安全策略只写在 system prompt 里。

---

## 13. 当前主流模型生态

> 模型版本变化很快，学习时不要死记版本号。更重要的是理解不同模型家族的定位、能力和使用约束。

### 13.1 闭源/商业模型

| 家族 | 常见定位 |
|---|---|
| OpenAI GPT 系列 | 通用推理、代码、工具调用、结构化输出、Agent 应用 |
| Anthropic Claude 系列 | 长文档、写作、复杂分析、代码、企业 Agent |
| Google Gemini 系列 | 多模态、长上下文、Google Cloud/Workspace 生态 |
| xAI Grok 系列 | 实时信息、对话和平台生态 |
| Cohere Command 系列 | 企业 RAG、多语言、检索增强 |
| Mistral Commercial | 欧洲企业场景、开放与商业混合生态 |

### 13.2 开源/开放权重模型

| 家族 | 关注点 |
|---|---|
| Llama | 开放权重生态、多模态、MoE、开发者社区 |
| Qwen | 多语言、代码、数学、开源生态、dense/MoE |
| DeepSeek | MoE、推理模型、RL、成本效率 |
| Mistral / Mixtral | MoE、欧洲开源生态、部署友好 |
| Gemma | Google 开放模型生态 |
| Phi | 小模型、端侧、成本敏感场景 |
| Yi / GLM / Baichuan / MiniMax 等 | 中文与多语言场景可关注 |

### 13.3 模型选型原则

| 场景 | 优先考虑 |
|---|---|
| 复杂推理/代码 | frontier reasoning model |
| 简单分类/抽取 | 小模型或便宜模型 |
| 企业知识库 | RAG + 可靠引用，不一定要最大模型 |
| 高并发客服 | 小模型路由 + 缓存 + 大模型兜底 |
| 私有化部署 | 开放权重模型 + vLLM/TGI/TensorRT-LLM |
| 端侧运行 | 量化小模型 + llama.cpp/GGUF |
| 合规敏感 | 私有部署、日志脱敏、权限隔离 |

---

## 14. 24 周系统学习计划

### 总体节奏

每周投入建议：

- 轻量：6-8 小时/周。
- 标准：10-15 小时/周。
- 强化：20 小时+/周。

每周固定产出：

1. 一页学习笔记。
2. 一个小实验或代码片段。
3. 一个评测样例。
4. 一个复盘：本周学会什么、哪里没懂、下周怎么补。

---

### 第 1 阶段：基础与 Transformer（第 1-4 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 1 周 | Python/PyTorch/数学复盘 | 熟悉 Tensor、autograd、训练循环 | 手写一个分类模型训练脚本 |
| 第 2 周 | NLP 与 Tokenizer | 理解 token、embedding、语言模型 | 训练一个小 tokenizer，统计不同文本 token 数 |
| 第 3 周 | Attention 与 Transformer | 理解 Self-Attention、mask、位置编码 | 手写 mini Transformer block |
| 第 4 周 | 小型语言模型 | 理解 next-token prediction 和生成采样 | 训练一个字符级 GPT，并实现 generate |

阅读重点：

- Attention Is All You Need。
- Hugging Face LLM Course 前几章。
- Stanford CS224N Transformer 相关课程。

---

### 第 2 阶段：大模型原理与模型生态（第 5-8 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 5 周 | 预训练与 Scaling Law | 理解数据、参数、算力关系 | 写一页 scaling law 总结 |
| 第 6 周 | 架构演进 | GQA/MQA/MoE/RoPE/长上下文 | 对比 Llama/Qwen/DeepSeek 架构特点 |
| 第 7 周 | 推理与解码 | temperature、top-p、beam、KV cache | 写一个解码参数实验报告 |
| 第 8 周 | 模型选型 | 闭源/开源、小模型/大模型、成本/质量 | 设计一个模型路由策略 |

---

### 第 3 阶段：Prompt、API 与结构化输出（第 9-10 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 9 周 | Prompt & Context Engineering | 会写稳定 prompt，处理长上下文 | 做一个文档总结器 |
| 第 10 周 | Tool Calling / Structured Output | JSON Schema、函数调用、校验重试 | 做一个结构化信息抽取 API |

项目建议：

- 输入一段合同，输出 JSON：甲方、乙方、金额、期限、风险条款、证据句。
- 评测输出 JSON 合法率、字段准确率、引用准确率。

---

### 第 4 阶段：RAG（第 11-14 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 11 周 | 基础 RAG | chunk、embedding、vector search | 做一个本地知识库问答 |
| 第 12 周 | 高级检索 | hybrid search、rerank、query rewrite | 比较不同检索策略 |
| 第 13 周 | RAG 评测 | context precision/recall、faithfulness | 构建 50 条 golden set |
| 第 14 周 | Agentic/Graph/SQL RAG | 多跳、图谱、数据库查询 | 做一个多数据源问答系统 |

项目建议：

- 企业制度问答系统。
- 论文阅读助手。
- 财务指标查询助手。
- 法务合同条款检索助手。

验收指标：

- 检索 recall。
- 引用准确率。
- 答案事实一致性。
- 无答案时拒答率。
- 平均时延。
- 平均 token 成本。

---

### 第 5 阶段：Agent、MCP、A2A（第 15-17 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 15 周 | Agent 基础 | ReAct、planning、tool use | 做一个能查资料、调用计算器、生成报告的 Agent |
| 第 16 周 | Agent 工作流 | LangGraph / LlamaIndex workflow / OpenAI Agents SDK | 做一个可追踪多步骤工作流 |
| 第 17 周 | MCP/A2A | 标准化工具接入与多 Agent 协作 | 写一个 MCP server，并设计 A2A agent card |

项目建议：

- 研究助手：搜索、阅读、总结、引用、生成报告。
- 数据分析助手：读取 CSV、生成图表、解释结论。
- 工单处理助手：分类、查知识库、生成回复、升级人工。
- 代码助手：读取 repo、定位 bug、生成补丁、运行测试。

---

### 第 6 阶段：微调与后训练（第 18-20 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 18 周 | SFT 与数据构造 | chat template、loss mask、数据质量 | 构造 500 条 SFT 数据 |
| 第 19 周 | LoRA/QLoRA | PEFT、显存、adapter、merge | 微调一个 1B-7B 模型 |
| 第 20 周 | DPO/RLHF/GRPO | 偏好数据、奖励、可验证任务 | 做一个小规模 DPO 或 GRPO 实验 |

验收：

- 微调前后 eval 对比。
- 输出格式合法率。
- 任务成功率。
- 是否过拟合。
- 推理成本是否可接受。

---

### 第 7 阶段：推理部署、LLMOps 与安全（第 21-23 周）

| 周 | 主题 | 学习目标 | 产出 |
|---|---|---|---|
| 第 21 周 | 推理服务 | vLLM/TGI/llama.cpp/TensorRT-LLM | 部署一个 OpenAI-compatible 本地模型服务 |
| 第 22 周 | LLMOps | tracing、gateway、fallback、成本监控 | 做一个带监控的 LLM API 网关 |
| 第 23 周 | 安全与治理 | OWASP LLM、NIST AI RMF、红队 | 写一份安全测试清单并执行 |

---

### 第 8 阶段：Capstone 项目（第 24 周）

做一个完整项目，要求：

- 有真实业务场景。
- 有 RAG 或工具调用。
- 有结构化输出。
- 有 eval golden set。
- 有 tracing。
- 有权限或安全设计。
- 有部署方案。
- 有成本估算。
- 有失败案例分析。

推荐 Capstone：

1. **企业知识库 + 工单 Agent**  
   功能：文档问答、工单分类、引用、回复草稿、人工升级。  
   技术：RAG、rerank、Agent、eval、权限过滤。

2. **数据分析 Agent**  
   功能：读取数据库/CSV、生成 SQL、画图、解释指标。  
   技术：tool calling、SQL sandbox、structured output、human approval。

3. **代码仓库维护 Agent**  
   功能：理解 repo、定位 bug、生成 patch、运行测试。  
   技术：repo RAG、代码 Agent、shell tool、测试闭环。

4. **多模态文档审核系统**  
   功能：合同/票据/报告解析、风险识别、证据引用。  
   技术：OCR、VLM、RAG、JSON output、risk eval。

---

## 15. 12 周压缩版路线

如果时间有限，按这个学：

| 周 | 主题 |
|---|---|
| 1 | Python/PyTorch/Tokenizer/Transformer 快速复盘 |
| 2 | LLM API、Prompt、结构化输出 |
| 3 | RAG 基础：chunk、embedding、vector search |
| 4 | RAG 进阶：hybrid、rerank、query rewrite |
| 5 | RAG eval：golden set、Ragas、faithfulness |
| 6 | Tool calling、Agent 基础 |
| 7 | LangGraph/LlamaIndex Agent workflow |
| 8 | MCP 与多工具系统 |
| 9 | LoRA/QLoRA 微调 |
| 10 | vLLM/llama.cpp 部署与量化 |
| 11 | LLMOps、安全、成本优化 |
| 12 | Capstone 项目上线版 |

---

## 16. 推荐学习资源

### 16.1 免费课程与教程

| 资源 | 用途 | 链接 |
|---|---|---|
| Stanford CS224N | NLP 与 Transformer 理论基础 | [CS224N](https://web.stanford.edu/class/cs224n/) |
| Hugging Face LLM Course | Transformers、Datasets、Tokenizers、Accelerate | [HF LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/1) |
| Hugging Face Agents Course | Agent 理论与实践 | [HF Agents Course](https://huggingface.co/agents-course) |
| Hugging Face MCP Course | MCP 实战 | [HF MCP Course](https://huggingface.co/learn/mcp-course/en/unit0/introduction) |
| Full Stack LLM Bootcamp | LLM 应用全栈落地 | [FSDL LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) |
| DeepLearning.AI Advanced RAG | RAG 进阶与评测 | [Building and Evaluating Advanced RAG](https://www.deeplearning.ai/courses/building-evaluating-advanced-rag) |
| DeepLearning.AI LangGraph | Agent 工作流 | [AI Agents in LangGraph](https://www.deeplearning.ai/courses/ai-agents-in-langgraph) |
| DeepLearning.AI Agentic RAG | Agentic RAG | [Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/courses/building-agentic-rag-with-llamaindex) |

### 16.2 官方文档

| 类别 | 资源 |
|---|---|
| OpenAI API | [Models](https://developers.openai.com/api/docs/models)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning)、[Function Calling](https://developers.openai.com/api/docs/guides/function-calling)、[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[Agents SDK](https://developers.openai.com/api/docs/guides/agents) |
| Anthropic Claude | [Claude Docs](https://platform.claude.com/docs/en/home) |
| Google Gemini | [Gemini API Models](https://ai.google.dev/gemini-api/docs/models) |
| MCP | [Model Context Protocol Docs](https://modelcontextprotocol.io/docs/getting-started/intro) |
| A2A | [A2A Protocol Docs](https://a2a-protocol.org/latest/) |
| LangChain/LangGraph | [LangGraph Agentic RAG](https://docs.langchain.com/oss/python/langgraph/agentic-rag) |
| LlamaIndex | [LlamaIndex RAG Introduction](https://developers.llamaindex.ai/python/framework/understanding/rag/) |
| Haystack | [Haystack Docs](https://docs.haystack.deepset.ai/docs/intro) |
| Ragas | [Ragas Docs](https://docs.ragas.io/en/stable/) |
| vLLM | [vLLM Docs](https://docs.vllm.ai/en/latest/) |
| Hugging Face TGI | [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index) |
| TensorRT-LLM | [NVIDIA TensorRT-LLM](https://docs.nvidia.com/tensorrt-llm/index.html) |
| llama.cpp | [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) |
| PEFT | [Hugging Face PEFT](https://huggingface.co/docs/peft/en/index) |
| TRL | [Hugging Face TRL](https://huggingface.co/docs/trl/index) |
| LiteLLM | [LiteLLM Docs](https://docs.litellm.ai/) |
| DSPy | [DSPy Docs](https://dspy.ai/) |

### 16.3 经典论文阅读顺序

不用一口气全读完。建议按主题读摘要、图、方法、实验，再对照代码或博客。

#### Transformer 与预训练

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [BERT](https://arxiv.org/abs/1810.04805)
3. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
4. [Training Compute-Optimal Large Language Models / Chinchilla](https://arxiv.org/abs/2203.15556)
5. [LLaMA](https://arxiv.org/abs/2302.13971)

#### RAG 与检索

1. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. [REALM](https://arxiv.org/abs/2002.08909)
3. [ColBERT](https://arxiv.org/abs/2004.12832)
4. [Self-RAG](https://arxiv.org/abs/2310.11511)

#### 对齐与后训练

1. [InstructGPT](https://arxiv.org/abs/2203.02155)
2. [Constitutional AI](https://arxiv.org/abs/2212.08073)
3. [LoRA](https://arxiv.org/abs/2106.09685)
4. [QLoRA](https://arxiv.org/abs/2305.14314)
5. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
6. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

#### 推理与高效训练

1. [FlashAttention](https://arxiv.org/abs/2205.14135)
2. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
3. [Speculative Decoding](https://arxiv.org/abs/2211.17192)
4. [DeepSpeed](https://arxiv.org/abs/2207.00032)

#### 近期模型报告

1. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
2. [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
3. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
4. [Llama 4 Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)

### 16.4 安全与治理

| 资源 | 用途 |
|---|---|
| [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | LLM 应用安全风险清单 |
| [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework) | AI 风险管理框架 |
| [NIST Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) | 生成式 AI 风险管理 |
| [HELM](https://crfm.stanford.edu/helm/) | 模型综合评测 |
| [MLCommons AILuminate](https://mlcommons.org/ailuminate/) | AI 安全评测基准 |

---

## 17. 工具栈推荐

### 17.1 应用开发

| 目标 | 推荐工具 |
|---|---|
| 快速调用模型 | OpenAI / Anthropic / Gemini SDK |
| 统一多模型 | LiteLLM |
| RAG 原型 | LlamaIndex / LangChain |
| RAG 生产 | Haystack / LlamaIndex / LangGraph + 自建组件 |
| Agent 工作流 | LangGraph / OpenAI Agents SDK / LlamaIndex Workflows |
| Prompt 优化 | DSPy / promptfoo |
| API 服务 | FastAPI |
| 前端 | Streamlit / Gradio / Next.js |
| 数据库 | PostgreSQL / Redis / Elasticsearch |
| 向量库 | FAISS / Qdrant / Milvus / pgvector / Weaviate |
| 文档解析 | Unstructured / LlamaParse / PyMuPDF / Tesseract / PaddleOCR |
| 监控 | LangSmith / OpenTelemetry / Grafana / Prometheus |

### 17.2 训练微调

| 目标 | 推荐工具 |
|---|---|
| 基础模型加载 | Transformers |
| LoRA/QLoRA | PEFT |
| SFT/DPO/GRPO | TRL |
| 简化微调 | Axolotl / Unsloth |
| 分布式训练 | Accelerate / DeepSpeed / FSDP / Megatron-LM |
| 数据集 | Hugging Face Datasets |
| 实验跟踪 | Weights & Biases / MLflow |
| 模型管理 | Hugging Face Hub / ModelScope |

### 17.3 推理部署

| 目标 | 推荐工具 |
|---|---|
| GPU 高吞吐 | vLLM |
| HF 生态推理 | Text Generation Inference |
| NVIDIA 极致优化 | TensorRT-LLM |
| 本地/端侧 | llama.cpp / Ollama |
| 多模型网关 | LiteLLM |
| 企业推理服务 | Triton Inference Server |
| 量化格式 | GPTQ / AWQ / GGUF / bitsandbytes |

---

## 18. 项目组合：从入门到专家

### 项目 1：最小聊天 API

目标：

- 调用一个商业模型。
- 支持 streaming。
- 支持 system prompt。
- 支持 JSON 输出。
- 记录 token 和成本。

技能：

- API 调用。
- prompt 管理。
- 日志与错误处理。

### 项目 2：结构化抽取系统

输入：合同、简历、邮件、客服记录。  
输出：严格 JSON。

要求：

- JSON Schema 校验。
- 失败重试。
- 引用证据。
- 统计字段准确率。

技能：

- structured output。
- 数据标注。
- eval。

### 项目 3：企业知识库 RAG

要求：

- 文档上传。
- chunk + embedding。
- hybrid search。
- rerank。
- 答案引用。
- 无答案拒答。
- RAGAS 或自建 eval。

技能：

- RAG 全流程。
- 检索评测。
- 产品化部署。

### 项目 4：Agentic RAG

要求：

- Agent 自动判断是否检索。
- 可以多次检索。
- 可以调用计算器或搜索。
- 最终输出引用和推理摘要。
- 有 step limit 和 tracing。

技能：

- Agent。
- workflow。
- 工具调用。
- 安全控制。

### 项目 5：MCP 工具生态

要求：

- 写 2-3 个 MCP server。
- 一个文件工具。
- 一个数据库只读工具。
- 一个业务 API 工具。
- 加权限和审计。

技能：

- MCP。
- 工具协议。
- 企业集成。

### 项目 6：LoRA 微调

要求：

- 构造 500-2000 条高质量数据。
- 微调一个开源模型。
- 对比 base vs fine-tuned。
- 部署 adapter。
- 分析失败样例。

技能：

- PEFT。
- SFT。
- eval。
- 部署。

### 项目 7：本地模型推理服务

要求：

- 用 vLLM 或 llama.cpp 部署模型。
- 提供 OpenAI-compatible API。
- 做量化对比。
- 记录吞吐、时延、显存。
- 做模型路由。

技能：

- 推理优化。
- 成本控制。
- 生产部署。

### 项目 8：安全红队测试

要求：

- 针对 RAG/Agent 做 prompt injection 测试。
- 测试 system prompt leakage。
- 测试越权工具调用。
- 测试敏感信息泄露。
- 形成安全报告。

技能：

- LLM security。
- red teaming。
- governance。

---

## 19. 能力自测清单

### 19.1 初级

你应该能回答：

- Transformer 为什么需要位置编码？
- temperature 和 top-p 有什么区别？
- 为什么 LLM 会幻觉？
- RAG 的 chunk size 怎么选？
- embedding 和 reranker 的区别是什么？
- 什么是 JSON Schema？
- 为什么 Agent 需要 step limit？

### 19.2 中级

你应该能回答：

- 长上下文和 RAG 如何取舍？
- 什么时候微调比 prompt 更合适？
- LoRA rank 怎么影响效果和显存？
- 如何构造 RAG golden set？
- 如何评估 citation correctness？
- Agent 工具调用失败如何重试？
- 如何防止 SQL Agent 越权？

### 19.3 高级

你应该能回答：

- GQA/MQA 如何减少 KV cache？
- MoE 的 router 和 load balancing 有什么问题？
- DPO 与 PPO/RLHF 的核心差别是什么？
- GRPO 为什么适合可验证推理任务？
- speculative decoding 为什么能加速？
- 如何设计多模型路由和 fallback？
- 如何建立企业级 LLM 风险治理体系？

---

## 20. 面试/实战题库

### 20.1 RAG 题

1. 给你 10 万份 PDF，如何构建知识库？
2. 如何处理表格、图片、脚注和跨页内容？
3. 用户问“公司年假规则是什么”，系统答错了，你怎么排查？
4. RAG 中 embedding recall 高但最终答案差，可能原因是什么？
5. 如何避免检索到用户无权查看的文档？
6. 如何评估 RAG 是否比直接长上下文更好？

### 20.2 Agent 题

1. Agent 与普通 workflow 的区别是什么？
2. 如何设计一个不会无限循环的 Agent？
3. 工具调用参数由模型生成，如何保证安全？
4. 多 Agent 系统如何避免互相甩锅？
5. MCP server 被 prompt injection 攻击怎么办？
6. 什么时候不应该用 Agent？

### 20.3 微调题

1. SFT 数据如何构造？
2. 为什么微调后模型反而变差？
3. LoRA、QLoRA、全量微调如何选择？
4. 偏好数据如何采集？
5. DPO 为什么不需要显式 reward model？
6. 如何判断微调是否值得？

### 20.4 推理部署题

1. prefill 和 decode 的瓶颈分别是什么？
2. KV cache 为什么占显存？
3. PagedAttention 解决了什么问题？
4. continuous batching 如何提升吞吐？
5. int4 量化会带来什么风险？
6. 如何设计线上多模型路由？

---

## 21. 每日/每周学习方法

### 每日 45 分钟

- 15 分钟读文档/论文。
- 20 分钟写代码或跑实验。
- 10 分钟记录问题和结论。

### 每周 3 小时

- 1 小时复盘本周知识。
- 1 小时做一个小实验。
- 1 小时整理成一篇笔记或 demo。

### 每月

- 做一个完整小项目。
- 参加一次 benchmark/eval 对比。
- 更新一次技术雷达。
- 清理一次“只会概念不会代码”的知识点。

---

## 22. 推荐关注的信息源

| 类型 | 来源 |
|---|---|
| 趋势报告 | Stanford AI Index、State of AI Report |
| 论文 | arXiv、Papers with Code、Hugging Face Papers |
| 开源模型 | Hugging Face、ModelScope、GitHub trending |
| 框架更新 | OpenAI、Anthropic、Google AI、Hugging Face、LangChain、LlamaIndex、vLLM |
| 安全 | OWASP、NIST、MLCommons、Trail of Bits、Promptfoo blog |
| 中文社区 | 机器之心、量子位、智源、开源项目社区 |
| 实战 | Full Stack Deep Learning、DeepLearning.AI、Hugging Face Learn |

---

## 23. 学习路线分支

### 23.1 想做大模型应用工程师

重点：

- Prompt。
- RAG。
- Agent。
- API。
- 工具调用。
- LLMOps。
- Eval。
- 安全。

可以少学：

- 大规模预训练。
- 复杂分布式训练。
- 深度 RL 理论。

### 23.2 想做大模型算法工程师

重点：

- Transformer。
- 预训练。
- 数据工程。
- SFT/LoRA/DPO/RLHF/GRPO。
- 推理优化。
- 模型评测。
- 论文复现。

必须加强：

- 数学。
- PyTorch。
- 分布式训练。
- GPU 性能。

### 23.3 想做 AI 产品/解决方案架构师

重点：

- 能力边界。
- 成本估算。
- RAG/Agent 方案设计。
- 评测指标。
- 安全合规。
- 用户体验。
- 业务流程改造。

必须能判断：

- 哪些需求适合大模型。
- 哪些需求不能只靠 prompt。
- 哪些场景必须 human-in-the-loop。
- 哪些功能风险太高不能自动化。

---

## 24. 最后建议

1. **不要只看论文不做项目。** 大模型能力很大一部分来自工程细节。
2. **不要只追最新模型。** 模型会变，RAG、Agent、Eval、部署、安全这些能力更长期。
3. **不要相信单次 demo。** 没有 eval 的 demo 不可信。
4. **不要把安全写进 prompt 就完事。** 权限、工具、数据、审计必须在系统层做。
5. **不要过早微调。** 先试 prompt、RAG、工具、路由和评测。
6. **不要忽视成本。** token、时延、GPU、重试、长上下文都会吞预算。
7. **一定要做完整闭环。** 数据 → 模型/提示 → 应用 → 评测 → 监控 → 反馈 → 改进。

---

## 25. 一句话路线

如果只能记一句：

> 先用 Prompt、RAG、Tool Calling 和 Eval 做出可靠应用；再用 Agent、MCP、LLMOps 和安全治理让它可上线；最后用 LoRA/DPO/GRPO、蒸馏、量化和推理优化把质量、成本和规模做到生产级。

