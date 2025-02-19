import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


class SiliconflowFactory:
    model_params = {
        "temperature": 0,
        "seed": 42,
    }

    @classmethod
    def get_model(cls, model_name: str):
        return ChatOpenAI(
            model=model_name,  # 模型名称
            openai_api_key=os.getenv("SILICONFLOW_API_KEY"),  # 在平台注册账号后获取
            openai_api_base="https://api.siliconflow.cn/v1",  # 平台 API 地址
            **cls.model_params,
        )

    @classmethod
    def get_default_model(cls):
        return cls.get_model(model_name="deepseek-ai/DeepSeek-V3")


if __name__ == "__main__":
    load_dotenv()
    llm = SiliconflowFactory.get_default_model()
    response = llm.invoke("你是谁？")
    print(response.content)
