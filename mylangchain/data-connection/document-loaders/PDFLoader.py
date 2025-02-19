from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
PyPDFLoader部分参数：
    file_path (str | PurePath) – The path to the PDF file to be loaded.
    headers   (dict | None) – Optional headers to use for GET request to download a file from a web path.
    password  (str | bytes | None) – Optional password for opening encrypted PDFs.
    mode      (Literal['single', 'page']) – The extraction mode, either “single” for the entire document or “page” for page-wise extraction.
    ...

"""
loader = PyPDFLoader(
    file_path="./llama2.pdf",
    # headers = None
    # password = None,
    mode="page",
)

docs = []

pages = loader.load_and_split()
print(f"总共加载了 {len(pages)} 页")

merged_docs = []

chunk_size = 15  # 每 chunk_size 页合并成一个文档对象
for i in range(0, len(pages), chunk_size):  # 从0开始，到len(pages)-1停止，步长为chunk_size
    # 合并多个页面的文本
    combined_text = ""
    for j in range(i, min(i + chunk_size, len(pages))):  # 处理小于 chunk_size 页的余数
        combined_text += pages[j].page_content + "\n"  # 将每页的文本拼接起来
    # 创建新的 Document 对象
    merged_docs.append(Document(page_content=combined_text.strip(),
                                metadata={"start_page": i + 1, "end_page": min(i + chunk_size, len(pages))}))
# 检查合并后的文档块数量
print(f"最终合并后文档块数量: {len(merged_docs)}")
# print(merged_docs)
# ############通过总页数来拆分文本


# 定义文本拆分器（约 1000 token 分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # 尽可能大，提高每次请求信息量
    chunk_overlap=100  # 适当增加重叠，保持上下文
)
# 处理 PDF 页面的文本
docs = text_splitter.split_documents(merged_docs)
print(f"拆分后共有 {len(docs)} 个文本块")
