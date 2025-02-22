from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory


# 输出结构
class SortEnum(str, Enum):
    mobile_data = 'mobile_data'
    price = 'price'


class OrderingEnum(str, Enum):
    ascend = 'ascend'
    descend = 'descend'


class Semantics(BaseModel):
    name: Optional[str] = Field(description="流量包名称", default=None)
    price_lower: Optional[int] = Field(description="价格下限", default=None)
    price_upper: Optional[int] = Field(description="价格上限", default=None)
    mobile_data_lower: Optional[int] = Field(description="流量下限", default=None)
    mobile_data_upper: Optional[int] = Field(description="流量上限", default=None)
    sort_by: Optional[SortEnum] = Field(description="按价格或流量排序", default=None)
    ordering: Optional[OrderingEnum] = Field(
        description="升序或降序排列", default=None)


if __name__ == "__main__":
    load_dotenv()

    parser = PydanticOutputParser(pydantic_object=Semantics)
    prompt = PromptTemplate(
        template="你是一个语义解析器。你的任务是将用户的输入解析成JSON表示。不要回答用户的问题。\n"
                 "用户输入:{text}\n"
                 "{format_instructions}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 模型
    model = SiliconflowFactory.get_default_model()

    # LCEL 表达式
    runnable = (
            {"text": RunnablePassthrough()} | prompt | model | parser
    )

    # 直接运行
    ret = runnable.invoke({"text": "不超过100元的套餐哪个流量最大"})

    print(ret)
