import uvicorn
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

model = OllamaLLM(model="deepseek-r1:14b")
prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
add_routes(
    app,
    prompt | model,
    path="/joke",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9999)
