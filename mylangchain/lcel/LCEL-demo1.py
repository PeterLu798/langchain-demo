from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory

load_dotenv()

prompt = ChatPromptTemplate.from_template("请将一个关于 {topic} 的笑话")

model = SiliconflowFactory.get_default_model()
output_parser = StrOutputParser()

# 这里就是 LCEL
chain = prompt | model | output_parser

print(chain.invoke({"topic": "杰瑞"}))
