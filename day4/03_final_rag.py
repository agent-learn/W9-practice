import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 解决模型下载慢
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

llm=ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)

# ================= 1. 准备多源合并数据 =================
docs=[
    Document(page_content="LangGraph 的 Checkpointer 机制用于持久化。",metadata={"category":"framework","year":2024}),
    Document(page_content="生产环境推荐使用 SqliteSaver 或 PostgresSaver。",metadata={"category":"framework","year":2024}),
    Document(page_content="Python 3.12 引入了新的类型参数语法。",metadata={"category":"language","year":2023}),
    Document(page_content="需要查询 Art_1062 这个型号的技术参数。",metadata={"category":"hardware","year":2024}),
]

vectorstore=Chroma.from_documents(docs,embeddings)

# ================= 2. 组装双路检索器 =================
# 路 1：BM25
bm25=BM25Retriever.from_documents(docs,k=5)
# 路 2：向量
vector_ret=vectorstore.as_retriever(search_kwargs={"k":5})
# 混合
hybrid=EnsembleRetriever(
    retrievers=[bm25,vector_ret],
    weights=[0.5,0.5]
)

# ================= 3. 组装 Reranker =================
cross_encoder=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker=CrossEncoderReranker(model=cross_encoder,top_n=3)
# 终极检索器
final_retriever=ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid,
)

# ================= 4. 组装最终生成 =================
def ask_rag(query:str):
    print(f"\n🔍 用户提问: {query}")

    # 第一步：检索并精排（拿回最精准的 Top-3）
    retrieved_docs=final_retriever.invoke(query)
    print(f"📦 精排召回文档数: {len(retrieved_docs)}")

    # 第二步：拼接上下文
    context="\n\n".join([f" 【资料 {i}】\n{d.page_content} （来源：{d.metadata}）" for i,d in enumerate(retrieved_docs,1)])

    # 第三步：构造 Prompt
    prompt=f"""你是一个专业的技术助手。请严格根据以下参考资料回答用户问题。
如果资料中没提到，请直接说“不知道”，不要自己编造。

参考资料：
{context}

用户问题：{query}
"""

    # 第四步：大模型生成
    print("🧠 大模型思考中...")
    response=llm.invoke(prompt)
    print(f"\n🤖 最终回答:\n{response.content}")

# ================= 测试 =================
if __name__ == "__main__":
    ask_rag("生产环境怎么持久化？")
    print("="*60)
    ask_rag("Art_1062 是什么？")
