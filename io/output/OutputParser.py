from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field


# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="纪年: BC or AD")


llm = OllamaLLM(model="deepseek-r1:14b")

parser = JsonOutputParser(pydantic_object=Date)

prompt = PromptTemplate(
    template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

query = "2023年四月6日天气晴..."

input_prompt = prompt.format_prompt(query=query)
output = llm.invoke(input_prompt)
print("原始输出:\n" + output)

print("\n解析后:")
print(parser.invoke(output))
