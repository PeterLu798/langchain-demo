from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

llm = ChatOllama(model="deepseek-r1:14b", temperature=0)


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: int = Field(description="How funny the joke is, from 1 to 10")


structured_llm = llm.with_structured_output(Joke, method="json_schema")
structured_llm.invoke("Tell me a joke about wombats")
