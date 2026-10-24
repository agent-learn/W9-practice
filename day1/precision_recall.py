ground_truth_keywords=["循环","条件分支"]

results=[
    {"content":"支持循环和条件边","is_relevant":True},
    {"content":"状态图基础","is_relevant":False},
    {"content":"也可以写条件分支","is_relevant":True},
]

top_k=len(results)
relevant_retrieved=sum(1 for r in results if r["is_relevant"])
precision=relevant_retrieved/top_k
print(f"Precision@{top_k} = {precision}")

total_relevant=2
recall=relevant_retrieved/total_relevant
print(f"Recall = {recall}")
