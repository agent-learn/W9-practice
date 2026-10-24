"""
RAG 检索自动评估脚本
"""
import os
from dotenv import load_dotenv
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# 准备测试集：(查询, 预期的正确关键词)
TEST_CASES = [
    {
        "query": "如果我要把多个节点并发执行的结果追加到一个列表中，而不是互相覆盖，应该怎么做？",
        "must_have_all": ["Annotated", "operator.add"]  # 必须同时包含
    },
    {
        "query": "我想在工具执行失败时，不要让整个流程崩溃，而是让大模型看到报错并自行纠正，这是 ToolNode 的默认行为吗？",
        "must_have_all": ["ToolMessage", "handle_tool_errors=False"] # 必须理解默认会捕获，不关就默认
    },
    {
        "query": "在生产环境中，MemorySaver 会导致重启丢失数据，官方推荐用什么替代它？",
        "must_have_all": ["SqliteSaver", "PostgresSaver"]
    },
    {
        "query": "我在节点里用 interrupt() 挂起了，人类审查完后，应该用什么命令传入参数并恢复执行？",
        "must_have_all": ["Command(resume=", "返回值"]
    }
]

long_text = """
LangGraph 进阶实战指南：构建可控、有状态的 AI 智能体

一、引言与基础概念
LangGraph 是由 LangChain 团队开发的一个开源库，专门用于构建有状态、多参与者的应用程序。与传统的线性 Chain 不同，LangGraph 允许开发者定义循环（Cycles）、条件分支（Conditional Edges）以及复杂的并行处理逻辑。它的核心思想是将 AI 工作流抽象为一张“图（Graph）”。在这张图中，节点（Node）代表具体的计算任务（比如调用大模型、执行数据库查询），而边（Edge）代表控制流，决定了任务之间的执行顺序。

二、核心组件：StateGraph 与 Reducer
在 LangGraph 中，最核心的类是 StateGraph。StateGraph 依赖于一个全局共享的 State（状态）对象。所有节点在运行时，都会接收当前 State，并返回一个字典，用于更新该 State。
这里需要特别注意 State 的定义方式。默认情况下，节点返回的字典会覆盖原 State 中的同名字段。但是，如果你希望某些字段能够不断追加（比如聊天记录），你需要使用 Annotated 配合 reducer 函数。例如，使用 `Annotated[list, operator.add]` 可以让多个节点在并发执行时，将各自的返回值追加到一个列表中，而不是互相覆盖。对于消息列表，LangGraph 提供了 `add_messages` 这个专门的 reducer。

三、持久化与记忆：MemorySaver 与 Checkpointer
如果没有持久化机制，每次执行图都会从零开始。为了让 Agent 能够记住历史对话或在长时间运行的任务中恢复，LangGraph 引入了 Checkpointer（检查点）机制。
最基础的 Checkpointer 是 MemorySaver，它将状态保存在内存中，适合开发和测试。但在生产环境中，进程一旦重启数据就会丢失，所以官方推荐使用 `SqliteSaver` 或 `PostgresSaver` 来实现持久化。
Checkpointer 必须与 `thread_id` 配合使用。在调用 `graph.invoke()` 时，在 `config={"configurable": {"thread_id": "xxx"}}` 中指定线程 ID。这样，同一个 thread_id 的多次调用将共享同一份状态历史。

四、Human-in-the-loop：人工介入与中断
在涉及高风险操作（如删除数据库、发送对外邮件）时，让大模型自动执行非常危险。LangGraph 提供了强大的 Human-in-the-loop（人机协同）能力。
实现中断有两种方式：
1. 静态中断：在编译图时，通过 `interrupt_before=["tools"]` 或 `interrupt_after=["agent"]` 指定在特定节点前后暂停。
2. 动态中断：在节点内部调用 `interrupt()` 函数。这个函数会立刻挂起图的执行，并将中断信息返回给调用者。调用者审查完毕后，通过传入 `Command(resume="同意")` 来恢复执行。恢复时，`Command` 中携带的值会作为 `interrupt()` 函数的返回值，继续往下执行。

五、工具调用：ToolNode 的高效执行
为了让大模型拥有调用外部工具的能力，LangGraph 提供了预置的 `ToolNode`。开发者只需要将定义好的工具列表（通过 `@tool` 装饰器定义）传入 `ToolNode`，它就会自动接管工具调用的解析、执行和结果返回。
`ToolNode` 默认支持并发执行。如果大模型在一条消息中同时请求调用多个工具，`ToolNode` 会利用多线程并发执行它们。此外，如果工具内部发生异常，`ToolNode` 默认会将报错信息捕获，并包装成 `ToolMessage` 返回给大模型，防止整个流程崩溃。如果不需要这种保护，可以在初始化时设置 `handle_tool_errors=False`。

六、流式输出：Streaming 模式
对于耗时较长的任务，一次性等待最终结果体验很差。LangGraph 支持多种流式输出模式（Stream modes）：
1. `values`：在每个步骤结束后，流式输出整个 State 的完整快照。
2. `updates`：仅输出每个节点返回的增量更新（字典）。这是最常用的模式。
3. `messages`：逐 Token 流式输出大模型生成的内容，体验最丝滑。

七、常见报错与避坑指南
1. `KeyError: 'xxx'`：通常是因为在节点中访问了 State 中未定义的字段，或者在没有传入初始状态的情况下直接运行。
2. `GraphRecursionError`：图的执行步数超过了默认的 `recursion_limit=25`，这通常意味着你的图出现了死循环，需要检查条件边的路由逻辑。
3. `ValueError: Function must have a docstring`：在定义工具时，如果没有使用 `@tool` 提供 description，系统会尝试读取函数的 docstring。如果都没有，就会报错。
"""

def get_chunks(chunk_size):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
    )
    return splitter.split_text(long_text)

embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)

def evaluate_chunk_size(chunk_size:int,top_k:int=1) -> Dict:
    chunks=get_chunks(chunk_size)
    if len(chunks)==0:
        return {
            "chunk_size":chunk_size,
            "precision":0,
            "recall":0,
            "mrr":0
        }

    vs=FAISS.from_texts(chunks,embeddings)

    total_hits=0
    total_reciprocal_rank=0.0
    total_relevant=0

    for test in TEST_CASES:
        query=test["query"]
        expected=test["must_have_all"]
        total_relevant+=len(expected)

        results=vs.similarity_search(query,k=top_k)

        hit_found=False
        for rank,doc in enumerate(results,start=1):
            if all(kw in doc.page_content for kw in expected):
                total_hits+=1
                total_reciprocal_rank+=1.0/rank
                hit_found=True
                break

    hit_rate=total_hits/len(TEST_CASES) if TEST_CASES else 0
    mrr=total_reciprocal_rank/len(TEST_CASES) if TEST_CASES else 0

    return {
        "chunk_size":chunk_size,
        "hit_rate":hit_rate,
        "mrr":mrr
    }

if __name__ == "__main__":
    chunk_sizes=[50,128,512]
    results=[]

    print("="*50)
    print("🚀 开始评估不同 Chunk 大小...")
    print("="*50)

    for size in chunk_sizes:
        print(f"\n正在评估 Chunk Size = {size} ...")
        metrics=evaluate_chunk_size(size,top_k=3)
        results.append(metrics)
        print(f"  Hit Rate (Recall@3): {metrics['hit_rate']:.2f}")
        print(f"  MRR: {metrics['mrr']:.2f}")

    # 输出最终对比表格
    print("\n" + "="*50)
    print("📊 最终评估报告")
    print("="*50)
    print(f"{'Chunk Size':<15} | {'Hit Rate':<10} | {'MRR':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['chunk_size']:<15} | {r['hit_rate']:<10.2f} | {r['mrr']:<10.2f}")
