from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import logging
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers").setLevel(logging.INFO)


# 准备文档和向量库 (同上午)
docs = [
    Document(page_content="LangGraph 通过 Checkpointer 实现持久化，在生产环境中推荐使用 SqliteSaver 或 PostgresSaver。", metadata={"source": "doc1"}),
    Document(page_content="MemorySaver 将状态保存在内存中，适用于开发和测试。", metadata={"source": "doc2"}),
    Document(page_content="Human-in-the-loop 可以通过 interrupt() 函数实现动态中断。", metadata={"source": "doc3"}),
]
embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)
vectorstore=Chroma.from_documents(docs,embeddings)

llm=ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
multi_query_retriever=MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k":2}),
    llm=llm,
)

query = "生产环境怎么持久化？"
print(f"原始 Query: {query}")
docs=multi_query_retriever.invoke(query)


print("\n检索结果:")
for doc in docs:
    print(f"  - {doc.page_content[:50]}...")
