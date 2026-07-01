下面是这个 A3 作业网页的**完整要求总结**，我按你写报告和做系统时最容易对照的方式整理。

## 1. 作业主题与目标

作业名称是 **A3: Personalised Multimodal Agent System**，总分 **20 marks**。你需要设计、实现并评估一个基于自己知识库的**个性化多模态智能 Agent 系统**。系统需要使用 **LangGraph Agent**，结合 **LLM / VLM**、向量数据库、元数据、检索流程、工具调用和记忆/状态机制，回答与你个人知识库相关的问题。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

重点不是做一个“能聊天的 chatbot”，而是要体现你对系统设计的思考：为什么这样设计知识库、为什么这样检索、为什么需要 agent routing / memory / tools，以及这些设计是否真的提升了 retrieval quality、reasoning ability 和 user interaction。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

一句话理解：
**这不是普通应用开发，而是一个小型 systems research project。**

------

## 2. 系统必须包含的核心部分

你的系统必须集成以下内容：

1. **Personalised Knowledge Base**
   必须是你自己整理的、真实个性化的知识库，比如课程材料、项目文档、研究论文、旅行记录、食谱、购物记录、兴趣收藏等。不能只是随便找公开数据。
2. **At least two modalities**
   至少两种模态，例如：
   - Text + Image
   - Text + Audio transcript
   - Text + Chart
   - Image + Metadata
   - Document text + Figures
3. **Retrieval / Indexing Pipeline**
   需要有结构化检索方法，例如：
   - vector database
   - multimodal embeddings
   - separate indices
   - OCR / caption indexing
   - hybrid retrieval
   - ranking / fusion
4. **Agent Framework**
   需要用 agent workflow 组织系统步骤，例如：
   - query routing
   - retrieval planning
   - memory and state
   - tool selection
   - task decomposition
   - verification stages
5. **Quantitative Evaluation**
   必须有定量评估，不可以只展示几个聊天截图。你需要比较不同系统设计，并分析效果差异。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

------

## 3. 你需要有一个清晰的技术问题 / 设计假设

网页明确要求：系统必须超越基础 chatbot，并围绕一个清晰的 **technical question / design hypothesis / innovation point** 展开。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

可以类似这样：

- Text-only indexing 是否足够支持 multimodal QA？
- Image-only embedding 能否支持检索？
- Hybrid multimodal retrieval 是否优于 single-space retrieval？
- Agentic routing 是否能改善复杂查询？
- Memory 是否能帮助多轮个性化交互？
- 将 retrieval、planning、answering 分开是否有收益？

高分关键是：
**你不是简单实现功能，而是提出一个可验证的系统设计问题，然后用实验回答它。**

------

## 4. Evaluation 必须怎么做

### 4.1 必须设计 benchmark suite

你需要设计一个小型但结构化的测试集，而不是随便问几个问题。网页要求至少覆盖 **4 类 query families**：([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

| Query Family                                    | 含义                                         |
| ----------------------------------------------- | -------------------------------------------- |
| Factual Retrieval                               | 直接检索知识库中的事实，例如某个食谱需要多久 |
| Cross-Modal Retrieval                           | 需要跨模态信息，例如根据图片找对应文本信息   |
| Analytical / Multi-Hop Synthesis                | 需要结合多个证据进行综合推理                 |
| Conversational Follow-Up / Personalised Context | 多轮对话或依赖用户偏好/记忆的问题            |

每一类至少要有 **one test case**，并且要分析不同系统版本在哪些地方成功、哪些地方失败。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

------

### 4.2 必须包含的指标

你的 evaluation 至少要包含：

**一个质量指标**，例如：

- Recall@k
- Top-k retrieval accuracy
- MRR
- task success rate
- keyword match
- groundedness
- human judgement
- LLM-as-judge scoring

以及**一个效率 / 系统指标**，例如：

- latency
- number of tool calls
- token usage

也就是说，不能只说“回答更好”，你要用表格或实验结果证明。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

------

### 4.3 必须做的系统比较

网页明确要求必须比较：

1. **plain LLM / VLM vs final agent system**
2. **至少一个 final design 的 ablation**

可选 ablation 例子：

- text-only index vs image-only index
- caption-only vs caption + image embeddings
- no memory vs memory
- no router vs router
- fixed pipeline vs tool-based agent

为了拿高分，建议至少做 3 个系统版本：

| Variant                  | 用途                                              |
| ------------------------ | ------------------------------------------------- |
| V0 Plain LLM             | 作为无检索 baseline                               |
| V1 RAG without agent     | 证明 retrieval 的作用                             |
| V2 Final LangGraph Agent | 证明 routing / memory / tool orchestration 的作用 |

如果你想更稳，可以加：

| Variant                              | 用途        |
| ------------------------------------ | ----------- |
| V3 Final Agent without memory/router | 做 ablation |

------

## 5. 交付物要求

你需要提交两个东西，分别提交：

1. **source code zip**
2. **report pdf**

命名格式是：

```text
[StudentID_Name.xxx]
```

例如：

```text
Sxxxxxxx_NAME.zip
Sxxxxxxx_NAME.pdf
```

Report 最大 **4 pages**，允许 appendix。网页要求报告写得像一篇 short systems paper。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

------

## 6. Source Code Repository 需要包含什么

代码压缩包里至少要包含：

- source code
- installation instructions
- dependencies
- run instructions

也就是说，你的代码需要可复现。别人拿到你的 zip 后，应该知道怎么安装、怎么运行、怎么测试。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

建议代码结构类似：

```text
project/
  README.md
  requirements.txt
  app.py
  agent/
  retrieval/
  data/
  evaluation/
  results/
  agent_trace.log
```

------

## 7. Report 必须包含的内容

报告最多 4 页，建议包含以下部分：

1. **Problem Statement**
   说明你做什么系统，面向什么场景，核心技术问题是什么。
2. **Knowledge Base Description**
   说明你的知识库来源、模态、数据规模、字段/元数据设计。
3. **Retrieval Design**
   说明 embedding、index、metadata、fusion、reranking、OCR/caption 等设计。
4. **Agent Workflow**
   说明 LangGraph 节点、工具、状态、memory、routing、verification 如何工作。
5. **Experiments & Ablation Studies**
   说明测试集、query families、baseline、variants、metrics。
6. **Results & Failure Analysis**
   用表格展示结果，并解释成功、失败、trade-off。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

------

## 8. 评分标准：20 分怎么分

总共 5 个部分，每个 **4 marks**。

### 8.1 Problem Framing & Innovation — 4 marks

高分要求：

- 有清晰且有吸引力的 design hypothesis
- 有独立技术贡献
- 和 teaching demo 明显不同
- 创新点有意义、有动机、有分析

低分风险：

- 只是实现一个普通 chatbot
- 问题定义模糊
- 只是改了 demo 的表面内容

------

### 8.2 Knowledge Base & Retrieval Design — 4 marks

高分要求：

- 知识库真实、个性化、结构清楚
- 至少两种模态被有意义地整合
- retrieval / indexing 设计合理
- 有实验比较不同检索策略

低分风险：

- 知识库太随便
- 多模态只是形式上存在
- 检索设计没有解释
- 直接套模板

------

### 8.3 Agent Framework & Tool Orchestration — 4 marks

高分要求：

- agent workflow 设计清楚
- 工具调用、routing、memory、state handling 有实际作用
- LangGraph 不是摆设
- agent 明显优于简单 linear RAG pipeline

低分风险：

- 只是普通 retrieval → answer
- 没有真正 tool orchestration
- workflow 和 demo 太像

------

### 8.4 Quantitative Evaluation & Ablation — 4 marks

高分要求：

- 有严谨 evaluation
- 有多个 baseline 和 ablation
- 指标合适，结果清楚
- 能解释为什么某些设计更好或更差

低分风险：

- 只有聊天截图
- 没有定量结果
- 没有系统版本比较
- 失败分析很浅

------

### 8.5 Report, Code & Reproducibility — 4 marks

高分要求：

- 报告结构专业清楚
- 代码可运行、可复现
- 有图表、结果、failure analysis
- 有独立开发证据

低分风险：

- README 不清楚
- 代码跑不起来
- 报告缺结果或缺分析
- 没有复现说明

------

## 9. Academic Integrity / Demo 使用限制

可以参考 teaching demo 学习 LangGraph，但它**不是 submission template**。最终项目必须在以下方面体现原创性：

- problem framing
- knowledge base design
- multimodal representation
- retrieval strategy
- agent workflow
- evaluation methodology

网页特别强调，如果只是复制 teaching demo，可能直接 **zero marks**。可能被认为不原创的证据包括：代码相似、workflow 一样、report language 复制、prompt 复制、tool definitions 复制、缺少原创性解释等。([Infs4205](https://infs4205-7205.github.io/A3-Personalised-Multimodal-Chatbot/))

------

## 10. 满分导向的完成清单

你最终最好确保有这些内容：

| 模块       | 必须完成                                                  |
| ---------- | --------------------------------------------------------- |
| 主题       | 一个清晰的 personalised multimodal agent 场景             |
| 技术问题   | 一个明确 design hypothesis                                |
| 知识库     | 真实个人数据，至少两种模态                                |
| 检索       | vector DB + metadata，最好有 hybrid / multimodal / fusion |
| Agent      | LangGraph workflow，包含 routing / memory / tools / state |
| Evaluation | 4 类 query families                                       |
| Metrics    | 至少 1 个质量指标 + 1 个效率指标                          |
| Baseline   | plain LLM/VLM                                             |
| Ablation   | 至少一个去掉 memory/router/index 的版本                   |
| Report     | 4 页内，像 short systems paper                            |
| Code       | zip，含 dependencies、installation、run instructions      |
| Integrity  | 明确说明和 demo 的区别                                    |

------

最稳的作业定位可以写成：

> This project investigates whether an agentic multimodal retrieval workflow improves personalised question answering compared with plain LLM and non-agentic RAG baselines. The system uses a personalised multimodal knowledge base, combines text/image metadata retrieval, and applies LangGraph-based routing, memory, and verification to support factual, cross-modal, multi-hop, and conversational queries.

这个方向基本覆盖了网页最核心的要求。