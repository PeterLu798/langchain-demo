from langchain import hub

# 下载一个现有的 Prompt 模板
prompt = hub.pull("vyang/rag-stepback")

print(prompt)
