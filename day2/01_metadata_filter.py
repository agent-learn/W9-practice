#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os
import json
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


docs=[
    Document(page_content="LangGraph 支持循环和条件分支。",metadata={"category":"framework","year":2024,"author":"LangChain"}),
    Document(page_content="Python 3.12 引入了新的类型参数语法。",metadata={"category":"language","year":2023,"author":"Python"}),
    Document(page_content="LangGraph 的持久化通过 Checkpointer 实现。",metadata={"category":"framework","year":2024,"author":"LangChain"}),
    Document(page_content="Java 21 增加了虚拟线程支持。",metadata={"category":"language","year":2023,"author":"Java"}),
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
    #temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

def extract_filter(query:str) -> dict:
    prompt=f"""你是一个查询分析器。请从用户的查询中提取出元数据过滤条件。
用户的查询：{query}
元数据字段只有两个：
- category: 字符串（可选值：'framework', 'language'）
- year: 整数（年份）
请直接返回一个 JSON 格式的过滤器。例如：{{"year": 2024}}
如果没有任何过滤条件，返回 {{}}。不要输出任何其他文字。
"""
    response=llm.invoke(prompt).content.strip()
    try:
        if "```json" in response:
            response=response.split("```json")[1].split("```")[0].strip()
        return json.loads(response)
    except:
        return {}

def smart_search(query:str):
    print(f"🔍 查询: {query}")

    filter_dict=extract_filter(query)
    print(f"📦 大模型提取到的原始条件: {filter_dict}")

    query_filter=None
    if filter_dict:
        if len(filter_dict)>1:
            and_conditions=[{k:v} for k,v in filter_dict.items()]
            query_filter={"$and":and_conditions}
        else:
            query_filter=filter_dict

        print(f"📦 转换后的 Chroma 过滤条件: {query_filter}")

    if query_filter:
        results=vectorstore.similarity_search(query,k=2,filter=query_filter)
    else:
        results=vectorstore.similarity_search(query,k=2)

    for i,doc in enumerate(results,1):
        print(f"  [{i}] 内容: {doc.page_content}")
        print(f"      元数据: {doc.metadata}")

print("="*50)
smart_search("2024年发布的关于 LangGraph 的文档有哪些？")
print("\n" + "="*50)
smart_search("关于 Java 语言的文档")
