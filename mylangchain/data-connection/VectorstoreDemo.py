from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载文档
loader = PyMuPDFLoader("document-loaders/llama2.pdf")
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
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
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

# 检索 top-3 结果
retriever = db.as_retriever(search_kwargs={"k": 3})

docs = retriever.invoke("llama2有多少参数")

for doc in docs:
    print(doc.page_content)
    print("----")
