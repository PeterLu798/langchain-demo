from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

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
    token_counter=OllamaLLM(model="deepseek-r1:14b"),
)
# 从打印结果看出为了保留45个Token，它从“Hmmm let me think...”开始截取
print(msg1)
# 要保留 system prompt，则设置include_system=True

print("*" * 50)

msg2 = trim_messages(
    messages,
    max_tokens=45,
    strategy="last",
    token_counter=OllamaLLM(model="deepseek-r1:14b"),
    include_system=True,
    allow_partial=True,
)
print(msg2)
