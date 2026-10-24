import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ========== 1. 准备数据 ==========
docs=[
    Document(page_content="LangGraph 支持循环和条件分支。",metadata={"source":"doc1"}),
    Document(page_content="生产环境持久化推荐 SqliteSaver 或 PostgresSaver。",metadata={"source":"doc2"}),
    Document(page_content="MemorySaver 仅适用于开发和测试。",metadata={"source":"doc3"}),
    Document(page_content="Python 3.12 引入了新的类型参数语法。",metadata={"source":"doc4"}),
    Document(page_content="Java 21 增加了虚拟线程支持。",metadata={"source":"doc5"}),
]

# ========== 2. 构建混合检索器 ==========
bm25=BM25Retriever.from_documents(docs,k=5)

embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)
vectorstore=Chroma.from_documents(docs,embeddings)
vector=vectorstore.as_retriever(search_kwargs={"k":5})

hybrid=EnsembleRetriever(
    retrievers=[bm25,vector],
    weights=[0.5,0.5]
)

# ========== 3. 构建 Rerank 检索器 ==========
cross_encoder=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker=CrossEncoderReranker(model=cross_encoder,top_n=3)

final_retriever=ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid,
)

# ========== 4. 对比测试 ==========
def compare(query):
    print("=" * 60)
    print(f"🔍 Query: {query}")
    print("-" * 60)
    
    print("\n【混合检索原始结果（未精排）】")    
    raw=hybrid.invoke(query)
    for i,d in enumerate(raw,1):
        print(f"  {i}. [{d.metadata['source']}] {d.page_content}")

    print("\n【Reranker 精排后结果（Top-3）】")
    reranked=final_retriever.invoke(query)
    for i,d in enumerate(reranked,1):
        print(f"  {i}. [{d.metadata['source']}] {d.page_content}")
    print("\n")

# 测试 1：问一个需要精确语义匹配的问题
compare("生产环境怎么做持久化才可靠？")

# 测试 2：问一个带专有名词的问题
compare("LangGraph 支持哪些控制流？")
