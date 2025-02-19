from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

msg = filter_messages(messages, include_types="human")
print(msg)

print("*" * 50)

msg2 = filter_messages(messages, exclude_names=["example_user", "example_assistant"])
print(msg2)

print("*" * 50)

msg3 = filter_messages(messages, include_types=[HumanMessage, AIMessage], exclude_ids=["3"])
print(msg3)
