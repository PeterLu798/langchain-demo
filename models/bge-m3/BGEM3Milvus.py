from pymilvus import model

bge_m3_ef = model.hybrid.BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',  # Specify t`he model name
    device='cpu',  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False  # Whether to use fp16. `False` for `device='cpu'`.
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

docs_embeddings = bge_m3_ef.encode_documents(docs)

print("Doc Embeddings:", docs_embeddings)
print("Dense document dim:", bge_m3_ef.dim["dense"], docs_embeddings["dense"][0].shape)
print("Sparse document dim:", bge_m3_ef.dim["sparse"], list(docs_embeddings["sparse"])[0].shape)

"""
查询
"""

queries = ["When was artificial intelligence founded",
           "Where was Alan Turing born?"]

query_embeddings = bge_m3_ef.encode_queries(queries)

# Print embeddings
print("Query Embeddings:", query_embeddings)
# Print dimension of dense embeddings
print("Dense query dim:", bge_m3_ef.dim["dense"], query_embeddings["dense"][0].shape)
# Since the sparse embeddings are in a 2D csr_array format, we convert them to a list for easier manipulation.
print("Sparse query dim:", bge_m3_ef.dim["sparse"], list(query_embeddings["sparse"])[0].shape)
