from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mylangchain.llms.siliconflow.Siliconflow import SiliconflowFactory

load_dotenv()

# 加载文档
loader = PyMuPDFLoader("llama2.pdf")
pages = loader.load_and_split()

# 文档切分
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents(
    [page.page_content for page in pages[:4]]
)

# 灌库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", cache_folder=r"D:\appCache\huggingface")
db = Milvus.from_documents(  # or Zilliz.from_documents
    documents=texts,
    embedding=embeddings,
    connection_args={
        "uri": "http://localhost:19530",
        "token": "root:Milvus"
    },
    drop_old=True,  # Drop the old Milvus collection if it exists
)

# 检索 top-2 结果
retriever = db.as_retriever(search_kwargs={"k": 2})

# Prompt模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = SiliconflowFactory.get_default_model()

# Chain
rag_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | prompt
        | model
        | StrOutputParser()
)

ret = rag_chain.invoke("Llama 2有多少参数")
print(ret)
