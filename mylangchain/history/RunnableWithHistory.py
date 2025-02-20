from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory


def get_session_history(session_id):
    # 通过 session_id 区分对话历史，并存储在 sqlite 数据库中
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")


if __name__ == "__main__":
    load_dotenv()

    model = SiliconflowFactory.get_default_model()

    runnable = model | StrOutputParser()

    runnable_with_history = RunnableWithMessageHistory(
        runnable,  # 指定 runnable
        get_session_history,  # 指定自定义的历史管理方法
    )

    runnable_with_history.invoke(
        [HumanMessage(content="你好，我叫Peter")],
        config={"configurable": {"session_id": "peter"}},
    )

    ret = runnable_with_history.invoke(
        [HumanMessage(content="你知道我叫什么名字")],
        config={"configurable": {"session_id": "peter"}},
    )

    print(ret)

    print("*" * 50 + "分割线" + "*" * 50)

    ret1 = runnable_with_history.invoke(
        [HumanMessage(content="你知道我叫什么名字")],
        config={"configurable": {"session_id": "test"}},
    )
    print(ret1)
