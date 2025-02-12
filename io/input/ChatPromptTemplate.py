from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama import OllamaLLM

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "你是{product}的客服助手。你的名字叫{name}"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

llm = OllamaLLM(model="deepseek-r1:14b")

prompt = template.format_messages(
    product="AGI课堂",
    name="瓜瓜",
    query="你是谁"
)

print(prompt)

ret = llm.invoke(prompt)

print(ret)
