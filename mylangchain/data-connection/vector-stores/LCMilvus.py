from pymilvus import connections, db, MilvusClient

conn = connections.connect(host="127.0.0.1", port=19530)
database = db.create_database("my_database")

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)


class MilvusConnector:
    def __init__(self):
        pass
