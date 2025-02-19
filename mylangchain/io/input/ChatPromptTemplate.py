from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory

load_dotenv()

template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "你是{product}的客服助手。你的名字叫{name}"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

llm = SiliconflowFactory.get_default_model()

prompt = template.format_messages(
    product="电信",
    name="瓜瓜",
    query="你是谁"
)

ret = llm.invoke(prompt)

print(ret)
