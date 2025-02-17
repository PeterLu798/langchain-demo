import json

from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    RRFRanker,
)

"""
定义检索器
"""


class HybridRetriever:
    def __init__(self, uri, collection_name="hybrid", dense_embedding_function=None):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_function = dense_embedding_function
        self.use_reranker = True
        self.use_sparse = True
        self.client = MilvusClient(uri=uri)

    def build_collection(self):
        if isinstance(self.embedding_function.dim, dict):
            dense_dim = self.embedding_function.dim["dense"]
        else:
            dense_dim = self.embedding_function.dim

        tokenizer_params = {
            "tokenizer": "standard",
            "filter": [
                "lowercase",
                {
                    "type": "length",
                    "max": 200,
                },
                {"type": "stemmer", "language": "english"},
                {
                    "type": "stop",
                    "stop_words": [
                        "a",
                        "an",
                        "and",
                        "are",
                        "as",
                        "at",
                        "be",
                        "but",
                        "by",
                        "for",
                        "if",
                        "in",
                        "into",
                        "is",
                        "it",
                        "no",
                        "not",
                        "of",
                        "on",
                        "or",
                        "such",
                        "that",
                        "the",
                        "their",
                        "then",
                        "there",
                        "these",
                        "they",
                        "this",
                        "to",
                        "was",
                        "will",
                        "with",
                    ],
                },
            ],
        }

        schema = MilvusClient.create_schema()
        schema.add_field(
            field_name="pk",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=True,
            max_length=100,
        )
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535,
            analyzer_params=tokenizer_params,
            enable_match=True,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim
        )
        schema.add_field(
            field_name="original_uuid", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(
            field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64
        ),
        schema.add_field(field_name="original_index", datatype=DataType.INT32)

        functions = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        )

        schema.add_function(functions)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="IP"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_data(self, chunk, metadata):
        embedding = self.embedding_function([chunk])
        if isinstance(embedding, dict) and "dense" in embedding:
            dense_vec = embedding["dense"][0]
        else:
            dense_vec = embedding[0]
        self.client.insert(
            self.collection_name, {"dense_vector": dense_vec, **metadata}
        )

    def search(self, query: str, k: int = 20, mode="hybrid"):

        output_fields = [
            "content",
            "original_uuid",
            "doc_id",
            "chunk_id",
            "original_index",
        ]
        if mode in ["dense", "hybrid"]:
            embedding = self.embedding_function([query])
            if isinstance(embedding, dict) and "dense" in embedding:
                dense_vec = embedding["dense"][0]
            else:
                dense_vec = embedding[0]

        if mode == "sparse":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse_vector",
                limit=k,
                output_fields=output_fields,
            )
        elif mode == "dense":
            results = self.client.search(
                collection_name=self.collection_name,
                data=[dense_vec],
                anns_field="dense_vector",
                limit=k,
                output_fields=output_fields,
            )
        elif mode == "hybrid":
            full_text_search_params = {"metric_type": "BM25"}
            full_text_search_req = AnnSearchRequest(
                [query], "sparse_vector", full_text_search_params, limit=k
            )

            dense_search_params = {"metric_type": "IP"}
            dense_req = AnnSearchRequest(
                [dense_vec], "dense_vector", dense_search_params, limit=k
            )

            results = self.client.hybrid_search(
                self.collection_name,
                [full_text_search_req, dense_req],
                ranker=RRFRanker(),
                limit=k,
                output_fields=output_fields,
            )
        else:
            raise ValueError("Invalid mode")
        return [
            {
                "doc_id": doc["entity"]["doc_id"],
                "chunk_id": doc["entity"]["chunk_id"],
                "content": doc["entity"]["content"],
                "score": doc["distance"],
            }
            for doc in results[0]
        ]


dense_ef = BGEM3EmbeddingFunction()
standard_retriever = HybridRetriever(
    uri="http://localhost:19530",
    collection_name="milvus_hybrid",
    dense_embedding_function=dense_ef,
)

"""
插入数据
"""
path = "codebase_chunks.json"
with open(path, "r") as f:
    dataset = json.load(f)

is_insert = True
if is_insert:
    standard_retriever.build_collection()
    for doc in dataset:
        doc_content = doc["content"]
        for chunk in doc["chunks"]:
            metadata = {
                "doc_id": doc["doc_id"],
                "original_uuid": doc["original_uuid"],
                "chunk_id": chunk["chunk_id"],
                "original_index": chunk["original_index"],
                "content": chunk["content"],
            }
            chunk_content = chunk["content"]
            standard_retriever.insert_data(chunk_content, metadata)

"""
测试稀疏搜索
"""
# results = standard_retriever.search("create a logger?", mode="sparse", k=3)
# print(results)

"""
评估
现在我们已经将数据集插入 Milvus，可以使用密集搜索、稀疏搜索或混合搜索来检索前 5 个结果。
您可以更改mode 并对每个结果进行评估。
我们提出了 Pass@5 指标，包括检索每个查询的前 5 个结果并计算召回率。
"""


def load_jsonl(file_path: str):
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


dataset = load_jsonl("evaluation_set.jsonl")
k = 5

mode = "hybrid"

total_query_score = 0
num_queries = 0

for query_item in dataset:

    query = query_item["query"]

    golden_chunk_uuids = query_item["golden_chunk_uuids"]

    chunks_found = 0
    golden_contents = []
    for doc_uuid, chunk_index in golden_chunk_uuids:
        golden_doc = next(
            (doc for doc in query_item["golden_documents"] if doc["uuid"] == doc_uuid),
            None,
        )
        if golden_doc:
            golden_chunk = next(
                (
                    chunk
                    for chunk in golden_doc["chunks"]
                    if chunk["index"] == chunk_index
                ),
                None,
            )
            if golden_chunk:
                golden_contents.append(golden_chunk["content"].strip())

    results = standard_retriever.search(query, mode=mode, k=5)

    for golden_content in golden_contents:
        for doc in results[:k]:
            retrieved_content = doc["content"].strip()
            if retrieved_content == golden_content:
                chunks_found += 1
                break

    query_score = chunks_found / len(golden_contents)

    total_query_score += query_score
    num_queries += 1

print("Pass@5: ", total_query_score / num_queries)
