"""
Chunk 大小对比实验
"""
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ============ 1. 准备长文本 ============
# 模拟一份产品说明书
long_text = """
LangGraph 是一个用于构建有状态、多参与者应用程序的库。
它扩展了 LangChain 表达式语言，允许你协调多个链（或参与者）在复杂工作流中的交互。
LangGraph 的核心是状态图（StateGraph）。状态图由节点（Node）和边（Edge）组成。
节点代表计算步骤，边代表控制流。LangGraph 支持循环（Cycles）和条件分支（Conditional Edges）。
此外，LangGraph 内置了持久化（Persistence），通过 Checkpointer 可以保存和恢复状态，非常适合 Human-in-the-loop 场景。
LangGraph 还提供了 ToolNode，可以自动执行 LLM 调用的工具。相比于传统的 AgentExecutor，LangGraph 提供了更底层的控制。
""" * 5  # 重复5次以模拟长文档

# ============ 2. 定义切分策略 ============
def create_vectorstore(chunk_size:int):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        length_function=len,
    )
    chunks=splitter.split_text(long_text)
    print(f"📦 Chunk Size: {chunk_size} -> 切分出 {len(chunks)} 个块")

    embeddings=OpenAIEmbeddings(
        model="BAAI/bge-large-zh-v1.5",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )
    vectorstore=FAISS.from_texts(chunks,embeddings)
    return vectorstore,chunks

vs_256,chunks_256=create_vectorstore(256)
vs_512,chunks_512=create_vectorstore(512)
vs_1024,chunks_1024=create_vectorstore(1024)

# ============ 3. 检索对比 ============
query = "LangGraph 支持循环和什么分支？"

print("\n" + "="*50)
print(f"🔍 测试查询: {query}")
print("="*50)

for name,vs in [("256",vs_256),("512",vs_512),("1024",vs_1024)]:
    docs=vs.similarity_search(query,k=1)
    print(f"\n 【Chunk Size: {name}】")
    print(f"检索结果:\n{docs[0].page_content.strip()[:200]}...")
    