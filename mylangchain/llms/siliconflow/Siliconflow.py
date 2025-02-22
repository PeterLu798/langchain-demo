import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


class SiliconflowFactory:
    model_params = {
        "temperature": 0,  # 适用于需要确定性回答的场景，如程序代码生成、自动化文档撰写、数据分析等
        "seed": 42,  # 输出将完全可复现，每次运行都生成相同的结果
    }

    @classmethod
    def get_model(cls, model_name: str):
        return ChatDeepSeek(
            model=model_name,  # 模型名称
            api_key=os.getenv("SILICONFLOW_API_KEY"),  # 在平台注册账号后获取
            api_base="https://api.siliconflow.cn/v1",  # 平台 API 地址
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
