from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-r1:14b")

from langchain.schema import (
    AIMessage,  # 等价于OpenAI接口中的assistant role
    HumanMessage,  # 等价于OpenAI接口中的user role
    SystemMessage  # 等价于OpenAI接口中的system role
)

messages = [
    SystemMessage(content="你是AGIClass的课程助理。"),
    HumanMessage(content="我是学员，我叫卢柏江。"),
    AIMessage(content="欢迎！"),
    HumanMessage(content="我是谁")
]

ret = llm.invoke(messages)

print(ret)
