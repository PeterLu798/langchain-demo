from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM

prompt = PromptTemplate.from_template("讲个关于{topic}的笑话")

llm = OllamaLLM(model="deepseek-r1:14b")

runnable = (
        {"topic": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# 流式输出
for s in runnable.stream("小明"):
    print(s, end="", flush=True)
