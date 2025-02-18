import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",  # 模型名称
    openai_api_key=os.getenv("SILICONFLOW_API_KEY"),  # 在平台注册账号后获取
    openai_api_base="https://api.siliconflow.cn/v1",  # 平台 API 地址
)

response = llm.invoke("你是谁？")
print(response.content)
