from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,  # 等价于OpenAI接口中的assistant role
    HumanMessage,  # 等价于OpenAI接口中的user role
    SystemMessage  # 等价于OpenAI接口中的system role
)

"""
调用OpenAI
"""

llm = ChatOpenAI(model="gpt-4o-mini")  # 默认是gpt-3.5-turbo
response = llm.invoke("你是谁")
print(response.content)

"""
多轮对话
"""

messages = [
    SystemMessage(content="你是一个医生。"),
    HumanMessage(content="我叫Peter。"),
    AIMessage(content="欢迎！"),
    HumanMessage(content="我是谁")
]
ret = llm.invoke(messages)
print(ret.content)
