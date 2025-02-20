from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingFactory:
    model_kwargs = {

    }

    @classmethod
    def get_embedding_model(cls, model_name_or_path: str, cache_folder: str = None):
        """
        返回一个向量模型
        Args
        :param model_name_or_path: 模型名称或完整下载路径，如果只提供名称，则默认从HuggingFace下载
        :param cache_folder: 缓存目录
        :return:
        """
        return HuggingFaceEmbeddings(model_name=model_name_or_path, cache_folder=cache_folder, **cls.model_kwargs)

    @classmethod
    def get_default_embedding_model(cls, cache_folder: str):
        return cls.get_embedding_model(model_name_or_path="BAAI/bge-m3", cache_folder=cache_folder)
