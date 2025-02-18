from pymilvus import model

sentence_transformer_ef = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',  # Specify the model name
    device='cpu'  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

docs_embeddings = sentence_transformer_ef.encode_documents(docs)

print("Embeddings:", docs_embeddings)
print("Dim:", sentence_transformer_ef.dim, docs_embeddings[0].shape)

"""
查询
"""
queries = ["When was artificial intelligence founded",
           "Where was Alan Turing born?"]

query_embeddings = sentence_transformer_ef.encode_queries(queries)

# Print embeddings
print("Embeddings:", query_embeddings)
# Print dimension and shape of embeddings
print("Dim:", sentence_transformer_ef.dim, query_embeddings[0].shape)
