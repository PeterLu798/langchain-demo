import os

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.utils import ConfigurableField
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

# 模型1
ernie_model = QianfanChatEndpoint(
    qianfan_ak="xxx",
    qianfan_sk="xxx"
)

# 模型2
gpt_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key="xxx")

# 模型3
ollama_model = OllamaLLM(model="deepseek-r1:14b")

# 通过 configurable_alternatives 按指定字段选择模型
model = gpt_model.configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="gpt",
    ernie=ernie_model,
    ollama=ollama_model,
    # claude=claude_model,
)

# Prompt 模板
prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

# LCEL
chain = (
        {"query": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
)

# 运行时指定模型 "gpt" or "ernie"
ret = chain.with_config(configurable={"llm": "ollama"}).invoke("请自我介绍")

print(ret)
