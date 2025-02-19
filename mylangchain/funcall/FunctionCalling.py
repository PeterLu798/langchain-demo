from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory


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


if __name__ == "__main__":
    load_dotenv()

    llm = SiliconflowFactory.get_default_model()

    llm_with_tools = llm.bind_tools([add, multiply])

    query = "What is 3 * 12? Also, what is 11 + 49?"
    messages = [HumanMessage(query)]

    output = llm_with_tools.invoke(messages)

    # json.dumps: 把对象转换为json字符串，indent: 参数根据数据格式缩进显示，读起来更加清晰。
    # print(json.dumps(output.tool_calls, indent=4))

    # 回传 Function Call 的结果
    messages.append(output)

    available_tools = {"add": add, "multiply": multiply}

    for tool_call in output.tool_calls:
        selected_tool = available_tools[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    # 打印多轮历史消息
    # for message in messages:
    #     print(message.model_dump_json(indent=4))

    new_output = llm_with_tools.invoke(messages)
    print("大模型最终回复：")
    print(new_output.content)
