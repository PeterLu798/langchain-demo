from langchain import hub
from langchain.agents import create_self_ask_with_search_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_ollama import OllamaLLM

# 下载一个模板
self_ask_prompt = hub.pull("hwchase17/self-ask-with-search")
# print(self_ask_prompt.template)

search = SerpAPIWrapper(serpapi_api_key="XXX")

tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="搜素引擎",
        max_results=1
    )
]

llm = OllamaLLM(model="deepseek-r1:14b")

# self_ask_with_search_agent 只能传一个名为 'Intermediate Answer' 的 tool
agent = create_self_ask_with_search_agent(llm, tools, self_ask_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent_executor.invoke({"input": "冯小刚的老婆演过哪些电影，用中文回答"})
