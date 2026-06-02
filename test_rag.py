import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import jieba
import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 复用你已有的模型和数据库
embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="taxi_knowledge_base")
client = OpenAI(
    api_key="sk-uupfypofhmpnpbbabxsanbmguibjyqsepyulxrnqxmhuaecx",
    base_url="https://api.siliconflow.cn/v1"
)

# 准备测试集：5-10个你论文里能明确回答的问题
test_cases = [
    {"question": "论文题目是什么？", "must_contain": ["茶多酚", "超声处理", "发芽糙米"]},
    {"question": "作者姓名是什么？", "must_contain": ["张路坤"]},
    {"question": "DPPH清除率最高的是哪个组？", "must_contain": ["水热"]},
    {"question": "发芽条件是什么？", "must_contain": ["发芽"]},
    {"question": "指导教师是谁？", "must_contain": ["朱静"]},
]

def ask_rag(question):
    """你现有的检索+问答逻辑，简化版"""
    query_emb = embedding_model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    context = "\n\n".join(results["documents"][0])
    
    prompt = f"""严格基于参考内容回答。参考：{context}\n问题：{question}\n回答："""
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=512
    )
    return resp.choices[0].message.content

# 跑测试
print("=" * 50)
for i, case in enumerate(test_cases):
    answer = ask_rag(case["question"])
    passed = any(kw in answer for kw in case["must_contain"])
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"[{status}] Q{i+1}: {case['question']}")
    print(f"  回答: {answer[:100]}")
    print(f"  必须包含: {case['must_contain']}")
    print()
def llm_judge(question, answer, reference_context):
    """让LLM对回答打分"""
    judge_prompt = f"""你是一个严格的评估员。根据参考内容和用户问题，对回答打分。

参考内容：{reference_context}
用户问题：{question}
系统回答：{answer}

请从3个维度各打1-5分，输出JSON：
{{
  "relevance": 分数,
  "accuracy": 分数, 
  "completeness": 分数,
  "reason": "一句话说明扣分原因"
}}

评分标准：
- relevance：回答是否切题
- accuracy：事实是否正确（对照参考内容）
- completeness：是否完整回答了问题"""
    
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0, max_tokens=256
    )
    return resp.choices[0].message.content

# 对每个测试用例跑LLM Judge
for case in test_cases:
    answer = ask_rag(case["question"])
    query_emb = embedding_model.encode([case["question"]], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=3)
    context = "\n\n".join(results["documents"][0])
    
    judge_result = llm_judge(case["question"], answer, context)
    print(f"Q: {case['question']}")
    print(f"Judge: {judge_result}")
    print()