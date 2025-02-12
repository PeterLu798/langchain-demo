from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

template = PromptTemplate.from_template("给我讲个关于{subject}的笑话")
print("===Template===")
print(template)
print("===Prompt===")
print(template.format(subject='小明'))

llm = OllamaLLM(model="deepseek-r1:14b")
# 通过 Prompt 调用 LLM
ret = llm.invoke(template.format(subject='小明'))
# 打印输出
print(ret)
