from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")  # 默认是gpt-3.5-turbo
response = llm.invoke("你是谁")
print(response.content)