import calendar

import dateutil.parser as parser
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_ollama import OllamaLLM

# 下载一个现有的 Prompt 模板
react_prompt = hub.pull("hwchase17/react")

# print(react_prompt.template)

search = SerpAPIWrapper(serpapi_api_key="963f97d9efa23721d8359f4d11b64fc19ecb873d41066e6ee9e80dffcb71dd57")
tools = [
    Tool.from_function(
        func=search.run,
        name="Search",
        description="useful for when you need to answer questions about current events"
    ),
]


# 自定义工具


@tool("weekday")
def weekday(date_str: str) -> str:
    """Convert date to weekday name"""
    d = parser.parse(date_str)
    return calendar.day_name[d.weekday()]


tools += [weekday]

llm = OllamaLLM(model="deepseek-r1:14b")

# 定义一个 agent: 需要大模型、工具集、和 Prompt 模板
agent = create_react_agent(llm, tools, react_prompt)
# 定义一个执行器：需要 agent 对象 和 工具集
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行
agent_executor.invoke({"input": "2024年周杰伦的演唱会星期几"})
