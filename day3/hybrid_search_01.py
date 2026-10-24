import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# 1. 准备测试文档（故意加入专有名词，让BM25发挥作用）
docs = [
    Document(page_content="LangGraph 的 Checkpointer 机制用于持久化。", metadata={"source": "doc1"}),
    Document(page_content="生产环境推荐使用 SqliteSaver 或 PostgresSaver。", metadata={"source": "doc2"}),
    Document(page_content="Python 3.12 引入了新的类型参数语法。", metadata={"source": "doc3"}),
    Document(page_content="需要查询 Art_1062 这个型号的技术参数。", metadata={"source": "doc4"}),  # 专有名词
]

# 2. 构建 BM25 检索器（关键词检索）
bm25_retriever=BM25Retriever.from_documents(docs,k=3)

# 3. 构建向量检索器（语义检索）
embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)
vectorstore=FAISS.from_documents(docs,embeddings)
vector_retriever=vectorstore.as_retriever(search_kwargs={"k":3})

# 4. ⭐ 创建混合检索器
hybrid_retriever=EnsembleRetriever(
    retrievers=[bm25_retriever,vector_retriever],
    weights=[0.5,0.5],
)

# 5. 测试：查询包含专有名词，BM25 会帮大忙
query = "Art_1062 的技术参数在哪里？"
results=hybrid_retriever.invoke(query)
print(f"混合检索结果（共 {len(results)} 条）：")
for i,doc in enumerate(results,1):
    print(f"  [{i}] {doc.page_content} (来源：{doc.metadata['source']})")
