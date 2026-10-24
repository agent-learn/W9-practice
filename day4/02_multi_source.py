from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

docs_from_pdf=[Document(page_content="这是 PDF 里的合同条款。",metadata={"file_type":"pdf","author":"法务部"})]
docs_from_md=[Document(page_content="这是 Markdown 里的技术文档。",metadata={"file_type":"md","author":"研发部"})]
docs_from_web=[Document(page_content="这是网页爬取的最新新闻。",metadata={"file_type":"html","author":"未知"})]

# 1. 合并所有文档源
all_docs=docs_from_pdf+docs_from_md+docs_from_web

# 2. 规范化 Metadata（统一补充缺失字段）
for doc in all_docs:
    doc.metadata["source_system"]="internal_wiki"
    if "author" not in doc.metadata:
        doc.metadata["author"]="unknown"

# 3. 统一切片（长文档切分为小块）
splitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
splitted_docs=splitter.split_documents(all_docs)

# 4. 统一存入向量库
embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)
vectorstore=Chroma.from_documents(splitted_docs,embeddings)

print(f"✅ 成功合并并入库 {len(splitted_docs)} 个文本块。")
print("🔍 检查元数据统一性：")
for doc in splitted_docs:
    print(f"  - 内容：{doc.page_content} | 来源：{doc.metadata}")
    