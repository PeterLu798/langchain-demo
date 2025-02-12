import json

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


llm = ChatOllama(model="deepseek-llm:7b")

llm_with_tools = llm.bind_tools([add, multiply])

query = "3的4倍是多少?"
messages = [HumanMessage(query)]

output = llm_with_tools.invoke(messages)

print(json.dumps(output.tool_calls, indent=4))

# 回传 Function Call 的结果
messages.append(output)

available_tools = {"add": add, "multiply": multiply}

for tool_call in output.tool_calls:
    selected_tool = available_tools[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

new_output = llm_with_tools.invoke(messages)
for message in messages:
    print(json.dumps(message.dict(), indent=4, ensure_ascii=False))
print(new_output.content)
