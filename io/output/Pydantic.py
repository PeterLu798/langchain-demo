from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")


llm = ChatOllama(model="qwen2.5:7b", format="json", temperature=0)
# llm = OllamaFunctions(model="deepseek-r1:14b", format="json", temperature=0)

# 定义结构化输出的模型
structured_llm = llm.with_structured_output(Date)

template = """提取用户输入中的日期。
用户输入:
{query}"""

prompt = PromptTemplate(
    template=template,
)

query = "2023年四月6日天气晴..."
input_prompt = prompt.format_prompt(query=query)

ret = structured_llm.invoke(input_prompt)
print(ret)
