import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import jieba
import chromadb
import gradio as gr
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# 配置（API Key建议用环境变量，不要硬编码）
# ==========================================

client = OpenAI(
    api_key="sk-uupfypofhmpnpbbabxsanbmguibjyqsepyulxrnqxmhuaecx",
    base_url="https://api.siliconflow.cn/v1"
)
# ==========================================
# 加载模型
# ==========================================
print("正在加载Embedding模型...")
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("Embedding模型加载完成")

print("正在加载Rerank模型...")
reranker = CrossEncoder("BAAI/bge-reranker-base")
print("Rerank模型加载完成")

# ==========================================
# ChromaDB
# ==========================================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="taxi_knowledge_base")

# ==========================================
# 全局变量
# ==========================================
all_chunks = []
bm25 = None

# ==========================================
# 文档加载
# ==========================================
def load_text_file(file_path="data_taxi.txt", chunk_size=800, chunk_overlap=150):
    global all_chunks, bm25
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"知识库文件不存在：{file_path}")

    print("开始加载知识库...")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        if end < len(full_text):
            last_newline = chunk.rfind('\n')
            if last_newline > chunk_size // 2:
                chunk = full_text[start:start + last_newline]
                end = start + last_newline
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    all_chunks = [c for c in chunks if c]
    print(f"文档分块完成，共 {len(all_chunks)} 块")

    # 删除旧向量
    try:
        old_data = collection.get()
        ids = old_data.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            print(f"已删除旧向量 {len(ids)} 条")
    except Exception as e:
        print("清空旧向量失败：", e)

    # 生成向量
    print("开始生成向量...")
    embeddings = embedding_model.encode(
        all_chunks, show_progress_bar=True, normalize_embeddings=True
    ).tolist()
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    collection.add(ids=ids, documents=all_chunks, embeddings=embeddings)
    print("向量库写入完成")

    # BM25索引
    tokenized_corpus = [list(jieba.cut(chunk)) for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25索引构建完成")
    print("知识库加载成功")
  

# ==========================================
# 混合检索 + Rerank
# ==========================================
def hybrid_search(query, top_k=3):
    if bm25 is None:
        raise RuntimeError("BM25尚未初始化")

    # 向量检索
    query_embedding = embedding_model.encode(
        [query], normalize_embeddings=True
    ).tolist()
    vec_res = collection.query(query_embeddings=query_embedding, n_results=top_k)
    vec_docs = vec_res["documents"][0]

    # BM25检索
    token_query = list(jieba.cut(query))
    scores = bm25.get_scores(token_query)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    bm25_docs = [all_chunks[i] for i in top_idx]

    # 合并去重，直接返回
    results = []
    seen = set()
    for doc in vec_docs + bm25_docs:
        if doc not in seen:
            seen.add(doc)
            results.append(doc)
    return results[:top_k]
# ==========================================
# 问答
# ==========================================
def respond(message, history):
    if not message.strip():
        yield "请输入问题"
        return
    try:
        context_list = hybrid_search(message, top_k=3)
        if not context_list:
            yield "未检索到相关内容，请换个问题试试"
            return
        context_str = "\n\n------------------\n\n".join(context_list)
        prompt = f"""你是一个专业文档问答助手。

要求：
1. 只能引用参考内容中的信息
2. 不得补充、推测或改写不存在的信息
3. 保持原有专业术语
4. 回答时使用完整句子
5. 不得输出无意义英文字符
6. 如果内容不完整，直接回答“文档中无相关信息”

参考内容：
{context_str}

问题：{message}
回答："""

        stream = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
            stream=True
        )
        result = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
                yield result
    except Exception as e:
        yield f"错误：{str(e)}"

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    load_text_file(file_path="data_taxi.txt")
    demo = gr.ChatInterface(
        fn=respond,
        title="论文知识库问答",
        description="基于混合检索（向量 + BM25 + Rerank）"
    )
    demo.launch(server_name="127.0.0.1", server_port=7860)