from pymilvus import MilvusClient

client = None
res = None

try:
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )

    query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354,
                    0.9029438446296592]

    res = client.search(
        collection_name="my_collection",
        data=[query_vector],
    )
finally:
    # 释放内存
    client.release_collection(
        collection_name="my_collection"
    )
    # 关闭连接
    client.close()
