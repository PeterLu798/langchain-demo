from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_ollama import OllamaLLM

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory

load_dotenv()

# llm = SiliconflowFactory.get_default_model() # NotImplementedError: get_num_tokens_from_messages() is not presently implemented for model cl100k_base
llm = OllamaLLM(model="deepseek-r1:14b")

messages = [
    SystemMessage("you're a good assistant, you always respond with a joke."),
    HumanMessage("i wonder why it's called langchain"),
    AIMessage(
        'Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    HumanMessage("and who is harrison chasing anyways"),
    AIMessage(
        "Hmmm let me think.\n\nWhy, he's probably chasing after the last cup of coffee in the office!"
    ),
    HumanMessage("what do you call a speechless parrot"),
]

msg1 = trim_messages(
    messages,
    max_tokens=45,
    strategy="last",
    token_counter=llm,
    # include_system=True,  # 如果要保留 system prompt，则设置include_system=True
)
# 从打印结果看出为了保留45个Token，它从“Hmmm let me think...”开始截取
print(msg1)

print("*" * 20 + "分割线" + "*" * 20)

msg2 = trim_messages(
    messages,
    max_tokens=45,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=True,  # strategy="last"则不管最后一条消息长度是否超过max_tokens，都保留
)
print(msg2)
