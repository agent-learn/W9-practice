import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from hybrid_search_01 import hybrid_retriever

# 1. 加载 BGE Reranker 模型（本地运行，首次会下载）
cross_encoder=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
reranker=CrossEncoderReranker(model=cross_encoder,top_n=3)

# 2. 包装成“压缩检索器”：先混合召回，再精排
rerank_retriever=ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever,
)

# 3. 测试：比较重排前后的差异
query = "生产环境应该用什么来持久化？"
print("=== 重排前（混合检索 Top-3）===")
before=hybrid_retriever.invoke(query)
for i,doc in enumerate(before[:3],1):
    print(f"  [{i}] {doc.page_content}")

print("\n=== 重排后（Reranker 精排 Top-3）===")
after=rerank_retriever.invoke(query)
for i,doc in enumerate(after,1):
    print(f"  [{i}] {doc.page_content}")
    