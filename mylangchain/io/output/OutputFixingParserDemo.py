from dotenv import load_dotenv
from langchain.output_parsers import OutputFixingParser
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
    new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

    prompt = PromptTemplate(
        template="提取用户输入中的日期。\n用户输入:{query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    query = "2023年四月6日天气晴..."

    input_prompt = prompt.format_prompt(query=query)
    output = llm.invoke(input_prompt)  # 这里大模型的能力其实已经足够可以应对这种情况了，生成的结果就是正确的。
    # bad_output = output.replace("4", "四")

    print("修复之前:")
    try:
        print(parser.invoke(output.content))
    except Exception as e:
        print(e)

    print("修复之后:")
    print(new_parser.invoke(output.content))
