from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus import (
    MilvusClient, DataType
)
from pymilvus import WeightedRanker

"""
1、创建具有多个向量的 Collections
  1.1 定义 Schema
  1.2 创建索引
  1.3 创建 Collections
"""

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1000)
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=5)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="dense",
    index_name="dense_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128},
)

index_params.add_index(
    field_name="sparse",
    index_name="sparse_index",
    index_type="SPARSE_INVERTED_INDEX",  # Index type for sparse vectors
    metric_type="IP",  # Currently, only IP (Inner Product) is supported for sparse vectors
    params={"inverted_index_algo": "DAAT_MAXSCORE"},  # The ratio of small vector values to be dropped during indexing
)

client.create_collection(
    collection_name="hybrid_search_collection",
    schema=schema,
    index_params=index_params
)

"""
2、插入数据
"""

data = [
    {"id": 0, "text": "Artificial intelligence was founded as an academic discipline in 1956.",
     "sparse": {9637: 0.30856525997853057, 4399: 0.19771651149001523, },
     "dense": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, ]},
    {"id": 1, "text": "Alan Turing was the first person to conduct substantial research in AI.",
     "sparse": {6959: 0.31025067641541815, 1729: 0.8265339135915016, },
     "dense": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, ]},
    {"id": 2, "text": "Born in Maida Vale, London, Turing was raised in southern England.",
     "sparse": {1220: 0.15303302147479103, 7335: 0.9436728846033107, },
     "dense": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, ]}
]
res = client.insert(
    collection_name="hybrid_search_collection",
    data=data
)

"""
3、创建多个 AnnSearchRequest 实例  
由于参数limit 设置为 2，因此每个AnnSearchRequest 都会返回 2 个搜索结果。
在本例中，创建了 2 个AnnSearchRequest ，因此总共会返回 4 个搜索结果。
"""

# Who started AI research?
query_dense_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                      0.9029438446296592]

search_param_1 = {
    "data": [query_dense_vector],
    "anns_field": "dense",
    "param": {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    },
    "limit": 2
}
request_1 = AnnSearchRequest(**search_param_1)

query_sparse_vector = {3573: 0.34701499565746674}, {5263: 0.2639375518635271}
search_param_2 = {
    "data": [query_sparse_vector],
    "anns_field": "sparse",
    "param": {
        "metric_type": "IP",
        "params": {}
    },
    "limit": 2
}
request_2 = AnnSearchRequest(**search_param_2)

reqs = [request_1, request_2]

"""
4、配置 Rerankers 策略 
"""
# 例 1：使用WeightedRanker策略
# 注意WeightedRanker中提供的权重值总数应等于您之前创建的 AnnSearchRequest 实例数，
# 此例中创建了两个AnnSearchRequest实例：request_1和request_2，
# 密集向量字段dense的权重是0.8，稀疏向量字段sparse的权重是0.3
rerank = WeightedRanker(0.8, 0.3)

# 示例 2：使用 RRFRanker
ranker = RRFRanker(100)

"""
5、执行混合搜索 
"""

res1 = client.hybrid_search(
    collection_name="hybrid_search_collection",
    reqs=reqs,
    ranker=ranker,
    limit=2
)
for hits in res1:
    print("TopK results:")
    for hit in hits:
        print(hit)
