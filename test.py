import random

# 造1万条食品相关文档
templates = [
    "茶多酚浓度为{}%时，对{}的DPPH清除率为{}%",
    "超声处理{}min后，{}的峰值粘度变化为{}cP",
    "温度{}℃条件下，{}的水分含量为{}%",
    "添加{}g/kg的{}，产品保质期延长{}天",
    "采用{}方法提取{}，提取率为{}%",
    "在pH{}条件下，{}的溶解度为{}mg/mL",
    "经过{}处理，{}的硬度从{}降至{}",
    "{}的抗氧化活性在{}℃时达到最高值{}%",
    "发酵{}h后，{}中氨基酸含量增加{}%",
    "使用{}干燥方式，{}的色差值为{}",
]

concentrations = ["0.5", "1.0", "1.5", "2.0"]
materials = ["发芽糙米", "糙米淀粉", "米糠蛋白", "大豆蛋白", "燕麦", "玉米淀粉"]
temps = ["25", "37", "50", "60", "80", "100"]

documents = []
for i in range(10000):
    t = random.choice(templates)
    doc = t.format(
        random.choice(concentrations),
        random.choice(materials),
        random.randint(10, 99),
        # 根据模板不同，可能多几个参数
        random.choice(concentrations),
        random.choice(materials),
        random.randint(50, 500),
        random.choice(temps),
        random.choice(temps),
        random.choice(temps),
    )
    # 简单处理，确保格式对得上
    documents.append(f"文档{i+1}：{doc[:60]}")

print(f"生成了{len(documents)}条数据")
import chromadb

# 创建本地数据库（数据存当前目录的chroma_db文件夹）
client = chromadb.PersistentClient(path="./chroma_db")

# 创建一个集合（类似数据库的表）
collection = client.get_or_create_collection(
    name="food_research",
    metadata={"hnsw:space": "cosine"}  # 用余弦相似度
)

# 分批导入（1万条一次塞可能卡，分5批）
batch_size = 2000
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    ids = [f"doc_{j}" for j in range(i, i+len(batch))]
    collection.add(documents=batch, ids=ids)

print(f"导入了{collection.count()}条数据")
import time

# 测试5次取平均
queries = [
    "茶多酚对DPPH清除率的影响",
    "超声处理对淀粉粘度的作用",
    "高温条件下的抗氧化活性",
    "发酵对氨基酸含量的影响",
    "干燥方式对色泽的影响"
]

for q in queries:
    start = time.time()
    results = collection.query(query_texts=[q], n_results=5)
    elapsed = (time.time() - start) * 1000  # 毫秒
    
    print(f"\n查询：{q}")
    print(f"耗时：{elapsed:.1f}ms")
    print(f"Top1：{results['documents'][0][0][:50]}")