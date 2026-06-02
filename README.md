
# RAG 知识库问答系统

基于 RAG（检索增强生成）架构的个人知识库问答系统，支持多格式文档导入、增量加载、混合检索+Rerank精排、流式输出。

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    用户提问                           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│                  混合检索层                           │
│  ┌──────────────┐    ┌──────────────┐               │
│  │  向量检索     │    │  BM25检索     │               │
│  │ (bge-small-zh│    │ (jieba分词)   │               │
│  │ + ChromaDB)  │    │              │               │
│  └──────┬───────┘    └──────┬───────┘               │
│         └──────┬────────────┘                        │
│                ▼                                     │
│         合并去重(最多20条)                             │
│                ▼                                     │
│     CrossEncoder Rerank精排                          │
│     (bge-reranker-base)                             │
│                ▼                                     │
│         返回Top-K结果                                │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│                  LLM生成层                           │
│  System Prompt(参考内容) + 对话历史(5轮) + 用户问题    │
│  → Qwen2.5-7B-Instruct → 流式输出                    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│                Gradio Web界面                        │
│         本地访问 / share=True公网链接                  │
└──────────────────────────────────────────────────────┘

数据流：
  knowledge/ → 多格式读取 → 固定长度切分(800字符) → Embedding → ChromaDB
  增量加载：.processed.json跟踪文件mtime，未变化跳过，删除文件自动清理chunk
```

## 功能

- **多格式文档**：txt / md / docx / pdf / xlsx / xls / csv / xmind
- **增量加载**：已处理的文件自动跳过，新增/修改的文件自动更新，删除的文件自动清理
- **混合检索**：向量检索（语义匹配）+ BM25检索（关键词匹配），合并去重
- **Rerank精排**：CrossEncoder对候选结果二次排序，提升检索精度
- **对话历史**：保留最近5轮对话上下文
- **流式输出**：实时逐字返回，提升交互体验
- **公网分享**：share=True生成临时公网链接（72小时有效）

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/zhanglukun2907-ux/skills.git
cd skills

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API Key

编辑 `rag_chat.py`，将API Key替换为你自己的：

```python
client = OpenAI(
    api_key="sk-你的key",  # 替换为你的硅基流动API Key
    base_url="https://api.siliconflow.cn/v1"
)
```

获取API Key：[硅基流动](https://cloud.siliconflow.cn/) 注册后在API Key页面创建。

### 3. 放入文档

将你的文档放入 `knowledge/` 文件夹：

```bash
mkdir knowledge
cp 你的文档.txt knowledge/
cp 你的文档.docx knowledge/
# 支持格式：txt md docx pdf xlsx xls csv xmind
```

### 4. 启动

```bash
# 首次启动会自动下载Embedding和Rerank模型（约2GB）
HF_ENDPOINT=https://hf-mirror.com python rag_chat.py
```

启动成功后访问：
- 本地：http://127.0.0.1:7860
- 公网：终端输出的 `https://xxx.gradio.live` 链接

### 5. 评估

```bash
HF_ENDPOINT=https://hf-mirror.com python test_rag.py
```

## 项目结构

```
.
├── rag_chat.py              # 主程序（模型加载+检索+问答+界面）
├── test_rag.py              # 评估脚本（UnitTest+LLMJudge）
├── requirements.txt         # Python依赖
├── README.md                # 项目说明
├── knowledge/               # 知识库文档目录
│   └── .processed.json      # 增量加载跟踪文件（自动生成）
├── chroma_db/               # ChromaDB向量数据库（自动生成）
└── chat_log.csv             # 问答日志（自动生成）
```

## 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Embedding | BAAI/bge-small-zh-v1.5 | 中文向量模型，512维，轻量快速 |
| 向量数据库 | ChromaDB | 本地持久化，零配置 |
| 关键词检索 | BM25 (jieba分词) | 中文分词+传统信息检索 |
| Rerank | BAAI/bge-reranker-base | CrossEncoder精排，提升检索精度 |
| LLM | Qwen2.5-7B-Instruct | 硅基流动API，成本低响应快 |
| 前端 | Gradio ChatInterface | 快速搭建对话界面 |

## 评估体系

三层评估方法：

1. **UnitTest**：确定性测试，验证关键问题的回答是否包含预期关键词
2. **LLMJudge**：大模型自动评估（relevance/accuracy/completeness），建议用32B模型
3. **HumanEval**：人工验证，日常使用中持续标注和反馈

当前结果（51 chunk论文数据集）：

| 问题 | 预期关键词 | 结果 |
|------|-----------|------|
| 论文题目 | 茶多酚、超声处理、发芽糙米 | ✅ |
| 作者姓名 | 张路坤 | ✅ |
| 发芽条件 | 发芽 | ✅ |
| 指导教师 | 朱静 | ✅ |
| DPPH清除率 | 水热 | ❌ 数据源缺失 |

## 已知问题与优化方向

- [ ] PDF加密文件自动跳过（已加容错）
- [ ] 升级Embedding模型为bge-large-zh-v1.5（适合60万字以上知识库）
- [ ] 增加元数据过滤（按文件/类别筛选，提升大库检索精度）
- [ ] 增加Query Rewrite（用户模糊问题→补全后检索）
- [ ] LLMJudge换用Qwen2.5-32B-Instruct（7B做结构化输出不稳定）

