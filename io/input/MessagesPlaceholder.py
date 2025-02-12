from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import OllamaLLM

human_prompt = "Translate your answer to {language}."
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    # variable_name 是 message placeholder 在模板中的变量名
    # 用于在赋值时使用
    [MessagesPlaceholder("history"), human_message_template]
)

human_message = HumanMessage(content="Who is Elon Musk?")
ai_message = AIMessage(
    content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer"
)

messages = chat_prompt.format_prompt(
    # 对 "history" 和 "language" 赋值
    history=[human_message, ai_message], language="中文"
)

print(messages.to_messages())

llm = OllamaLLM(model="deepseek-r1:14b")

result = llm.invoke(messages)
print(result)
