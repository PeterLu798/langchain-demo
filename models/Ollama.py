from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-r1:14b")
response = llm.invoke("你是谁")
print(response)
