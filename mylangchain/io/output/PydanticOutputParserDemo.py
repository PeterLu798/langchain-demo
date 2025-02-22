from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory


# 定义你的输出对象
class Date(BaseModel):
    year: int = Field(description="Year")
    month: int = Field(description="Month")
    day: int = Field(description="Day")
    era: str = Field(description="BC or AD")


if __name__ == "__main__":
    load_dotenv()

    llm = SiliconflowFactory.get_default_model()

    parser = PydanticOutputParser(pydantic_object=Date)

    prompt = PromptTemplate(
        template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    query = "2023年四月6日天气晴..."

    input_prompt = prompt.format_prompt(query=query)
    print("Prompt输入")
    print(input_prompt)

    output = llm.invoke(input_prompt)
    print("原始输出:\n" + output.content)

    print("\n解析后:")
    data = parser.invoke(output.content)
    print(data)

    print(data.year)
    print(data.month)
    print(data.day)
    print(data.era)
