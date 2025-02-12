from langchain.output_parsers import OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field


# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="表示纪年，取值为 BC 或 AD，该值不能为空")


llm = OllamaLLM(model="deepseek-r1:14b")

parser = PydanticOutputParser(pydantic_object=Date)
new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

prompt = PromptTemplate(
    template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

query = "2023年四月6日天气晴..."

input_prompt = prompt.format_prompt(query=query)
output = llm.invoke(input_prompt)
# bad_output = output.replace("4", "四")

print("修复之前:")
try:
    print(parser.invoke(output))
except Exception as e:
    print(e)

print("修复之后:")
print(new_parser.invoke(output))
