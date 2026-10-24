import os
import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)

vectorstore=Chroma(
    collection_name="my_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

def generate_doc_id(content:str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def update_documents(new_docs:list[Document]):
    for doc in new_docs:
        doc.metadata["doc_id"]=generate_doc_id(doc.page_content)

    existing_data=vectorstore.get()
    existing_ids=set(existing_data["ids"]) if existing_data["ids"] else set()

    docs_to_add=[]
    ids_to_add=[]

    for doc in new_docs:
        doc_id=doc.metadata["doc_id"]
        if doc_id not in existing_ids:
            docs_to_add.append(doc)
            ids_to_add.append(doc_id)
        else:
            print(f"⏭️ 文档已存在，跳过: {doc.page_content[:20]}...")

    if docs_to_add:
        print(f"✅ 新增 {len(docs_to_add)} 篇文档...")
        vectorstore.add_documents(docs_to_add,ids=ids_to_add)
    else:
        print("🔄 没有新增文档。")

print("=== 第一次运行（首次插入） ===")
docs_batch1=[
    Document(page_content="LangGraph 的持久化通过 Checkpointer 实现。"),
    Document(page_content="生产环境推荐使用 SqliteSaver。")
]
update_documents(docs_batch1)

print("\n=== 第二次运行（包含已存在文档和新文档） ===")
docs_batch2=[
    Document(page_content="LangGraph 的持久化通过 Checkpointer 实现。"),
    Document(page_content="Python 3.12 引入了新语法。")
]
update_documents(docs_batch2)
