import os
import json
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_classic.retrievers import MultiQueryRetriever

# ⭐ 导入新版 Agent 构建器
from langchain.agents import create_agent 

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# ================= 1. 基础准备（保持不变） =================
docs=[
    Document(page_content="LangGraph 支持循环和条件分支。",metadata={"category":"framework","year":2024}),
    Document(page_content="LangGraph 通过 Checkpointer 实现持久化，生产环境推荐 SqliteSaver。",metadata={"category":"framework","year":2024}),
    Document(page_content="Python 3.12 引入了新的类型参数语法。",metadata={"category":"language","year":2023}),
    Document(page_content="Java 21 增加了虚拟线程支持。",metadata={"category":"language","year":2023}),
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

logging.basicConfig()
logging.getLogger("langchain_classic.retrievers").setLevel(logging.INFO)


# ================= 2. 定义工具（保持不变） =================
@tool
def search_by_metadata(query:str) -> str:
    """当用户的查询中包含明确的年份或类别（如：2024年、框架、语言）时，使用此工具。"""
    prompt=f"""从查询中提取过滤条件。查询：{query}
    字段：category ('framework', 'language'), year (整数)。
    只返回JSON，如 {{"year": 2024}}。无则返回 {{}}。"""
    response=llm.invoke(prompt).content.strip()

    try:
        if "```json" in response:
            response=response.split("```json")[1].split("```")[0].strip()
        filter_dict=json.loads(response)
    except:
        filter_dict={}

    query_filter=None
    if filter_dict:
        if len(filter_dict)>1:
            query_filter={"$and":[{k:v} for k,v in filter_dict.items()]}
        else:
            query_filter=filter_dict

    results=vectorstore.similarity_search(query,k=2,filter=query_filter)
    return "\n".join([f"内容：{d.page_content} | 元数据: {d.metadata}" for d in results])

multi_retriever=MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k":2}),
    llm=llm,
)

@tool
def search_by_semantic(query:str) -> str:
    """当用户的查询是口语化的、模糊的，或者询问原理、怎么做时，使用此工具。"""
    results=multi_retriever.invoke(query)
    return "\n".join([f"内容：{d.page_content} | 元数据：{d.metadata}" for d in results])

# ================= 3. 新版 Agent 构建 =================
tools=[search_by_metadata,search_by_semantic]

agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个智能检索助手。请根据用户的查询类型，选择合适的检索工具。"
)

# ================= 4. 测试 =================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 测试 1：包含 Metadata 条件的查询")
    print("=" * 60)
    result1=agent.invoke({"messages":[("user","2024年发布的关于 LangGraph 的文档有哪些？")]})
    print("\n最终回复:")
    print(result1["messages"][-1].content)

    print("\n" + "=" * 60)
    print("🤖 测试 2：模糊的语义查询")
    print("=" * 60)
    result2=agent.invoke({"messages":[("user","我想了解一下生产环境的持久化方案")]})
    print("\n最终回复:")
    print(result2["messages"][-1].content)
