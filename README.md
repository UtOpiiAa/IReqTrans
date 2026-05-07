# RE-LLM v2

铁路信号安全需求自然语言到 Lspec 形式化规范的自动翻译系统，基于检索增强生成（RAG）技术。

## 项目结构

```
rellm_v2/
├── config/                     # 配置模块
│   ├── __init__.py             # 配置数据类和加载逻辑
│   └── default.yaml            # 默认配置文件
│
├── rag/                        # RAG 核心模块
│   ├── __init__.py             # RAG 方法注册
│   ├── models/                 # RAG 方法实现
│   │   ├── no_rag.py           # 无检索（zero-shot）
│   │   ├── basic_rag.py        # 基础 RAG（一次检索 + 生成）
│   │   ├── token_rag.py        # TokenRAG（注意力+熵检测动态检索）
│   │   ├── fixed_length_rag.py # 固定长度触发检索
│   │   ├── fixed_sentence_rag.py # 句子边界触发检索
│   │   ├── flare_rag.py        # FLARE（低置信度检测动态检索）
│   │   └── dragin.py           # DRAG（幻觉检测+回溯检索）
│   ├── retriever/              # 检索引擎
│   │   ├── base.py             # 检索器基类
│   │   ├── bm25_retriever.py   # BM25 关键词检索
│   │   ├── embedding_retriever.py # Embedding 向量检索（纯 NumPy）
│   │   └── hybrid_engine.py    # 混合检索引擎（多列加权）
│   └── prompts/                # 提示词
│       ├── rag_prompt.py       # 用户提示词模板
│       ├── system_prompt.py    # 系统提示词构建器
│       └── template_adapter.py # 多模型提示词格式适配器
│
├── knowledge_extractor/         # 知识提取模块（独立模块）
│   ├── extractor.py            # 知识提取器（调用外部 LLM API）
│   └── merger.py               # 知识合并器（去重合并）
│
├── lspec_parser/                # LSpec 解析模块
│   ├── ast_nodes.py             # AST 节点定义
│   ├── lexer.py                 # 词法分析器
│   ├── parser.py                # 语法解析器
│   ├── batch_parse.py           # 批量解析脚本
│   └── merge_code.py            # 解析结果合并
│
├── evaluation/                  # 评测模块（独立顶级目录）
│   ├── evaluate.py              # 评测核心逻辑
│   ├── batch_evaluate.py        # 批量评测脚本
│   ├── analyze_results.py       # 结果分析和可视化
│   └── metrics/                 # 评测指标
│       ├── base.py              # 指标基类
│       ├── exact_match.py       # 精确匹配
│       ├── edit_similarity.py   # 编辑相似度
│       ├── token_f1.py          # Token F1
│       ├── code_bert.py         # CodeBERT 相似度
│       ├── codebleu.py          # CodeBLEU
│       └── syntax_correctness.py # 语法正确性
│
├── utils/                      # 工具类
│   ├── file_utils.py           # 文件读写（Excel/文本/JSON）
│   ├── llm_client.py           # LLM 客户端（本地模型加载+生成+注意力分析）
│   └── embedding_client.py     # Embedding 客户端（本地模型编码）
│
├── scripts/                    # 运行脚本
│   ├── run.py                  # 统一推理入口
│   ├── run_all_methods.py      # 运行所有 RAG 方法
│   ├── merge_results.py        # 合并分片结果
│   ├── extract_knowledge.py    # 知识提取入口
│   └── merge_knowledge.py      # 知识合并入口
│
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/default.yaml`，设置模型路径和知识库路径：

```yaml
llm:
  model_name: "/path/to/Qwen3-8B"    # 本地模型路径

embedding:
  model_name: "/path/to/Qwen3-Embedding-0.6B"

rag:
  knowledge_path: "knowledge/om_knowledge.xlsx"
```

### 3. 运行推理

```bash
# 单方法推理
python -m scripts.run --method BasicRAG --model-path /path/to/model

# TokenRAG（可调注意力阈值）
python -m scripts.run --method TokenRAG --model-path /path/to/model --attention-threshold-percentile 75

# FLARE
python -m scripts.run --method FlareRAG --model-path /path/to/model

# 并行分片执行
python -m scripts.run --method DRAG --model-path /path/to/model --task-id 0 --total-tasks 4
python -m scripts.run --method DRAG --model-path /path/to/model --task-id 1 --total-tasks 4
```

### 4. 运行所有方法

```bash
python -m scripts.run_all_methods --model-path /path/to/model
```

### 5. 知识提取

```bash
# 先在 config/default.yaml 中配置 extraction.api
python -m scripts.extract_knowledge

# 合并提取的知识
python -m scripts.merge_knowledge
```

### 6. LSpec 解析

```bash
# 批量解析 LSpec 文件
python -m lspec_parser.batch_parse --input-dir /path/to/lspec_files --output-dir /path/to/output
```

### 7. 评测

```bash
# 单文件评测
python -m evaluation.batch_evaluate --results_dir /path/to/results --output eval_results.xlsx

# 多方法比较
python -m evaluation.batch_evaluate --results_dir /path/to/results --batch_dirs

# 分析评测结果
python -m evaluation.analyze_results analyze --input eval_results.xlsx

# 方法比较
python -m evaluation.analyze_results compare --dir /path/to/eval_results/
```

## 支持的 RAG 方法

| 方法 | 说明 |
|------|------|
| NoRAG | 无检索，纯 LLM 推理 |
| BasicRAG | 一次检索 + 生成 |
| TokenRAG | 基于 Token 级别的注意力+熵检测动态检索 |
| FixedLengthRAG | 每隔固定长度触发检索 |
| FixedSentenceRAG | 遇到句子边界（分号）触发检索 |
| FlareRAG | FLARE 低置信度检测动态检索 |
| DRAG | 幻觉检测 + 回溯重新检索 |

## 评测指标

| 指标 | 说明 |
|------|------|
| ExactMatch | 精确匹配率 |
| EditSimilarity | 编辑相似度（Levenshtein） |
| TokenF1 | Token 级别 F1 分数 |
| CodeBERTScore | CodeBERT 语义相似度（需安装 code_bert_score） |
| CodeBLEU | CodeBLEU 代码相似度（需安装 codebleu） |
| SyntaxCorrectness | 语法正确性检测 |

## 配置说明

所有配置集中在 `config/default.yaml`，主要配置项：

- **llm**: LLM 模型路径、温度、最大 token 数
- **embedding**: Embedding 模型配置
- **rag**: 检索参数（top_k、权重、幻觉阈值等）
- **extraction**: 知识提取 API 配置
- **output**: 输出目录

## 与旧版本的差异

1. **统一配置**: 所有硬编码路径和参数集中到 YAML 配置文件
2. **模块独立**: 知识提取和 RAG 推理分为两个独立模块
3. **工具统一**: FileOperator/LLMClient/EmbeddingClient 整合到 `utils/`
4. **命名规范**: 文件名使用 snake_case，类名使用 PascalCase
5. **去除冗余**: 删除未使用的 `llmClient/`（OpenAI API 客户端）、`_old` 方法等
6. **清晰入口**: 所有运行脚本统一在 `scripts/` 下，使用 `python -m scripts.xxx` 运行
