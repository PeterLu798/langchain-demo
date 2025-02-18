"""
bge-m3原生调用案例
需要安装的依赖：pip install -U FlagEmbedding

"""

"""
生成密集向量

BGEM3FlagModel 类参数详解：  
    model_name_or_path               (str): 模型名字或下载地址。如果提供了下载地址，那么会按照地址去下载，如果没有默认从HuggingFace Hub拉去
    normalize_embeddings             (bool, optional): True/False, 是否为标准稠密向量，默认True
    use_fp16                         (bool, optional): True/False, 如果为True则利用半精度浮点技术来加快计算速度，提高计算性能，默认True
    query_instruction_for_retrieval: (Optional[str], optional): 当加载你微调后的模型时，如果你没有在训练的json文件中为query添加指令，则将其设置为空字符串;
                                                                如果你在训练数据中为query添加了指令，更改为你新设置的指令。
    query_instruction_format:        (str, optional): The template for :attr:`query_instruction_for_retrieval`. Defaults to :data:`"{}{}"`.
    devices                          (Optional[Union[str, int, List[str], List[int]]], optional): 用CPU还是GPU来推理或训练模型，
                                     例如：使用 CPU：devices="cpu"，使用第一个 GPU：devices="cuda:0"，使用多个GPU：devices=["cuda:0", "cuda:1"]
    pooling_method                   (str, optional): Pooling method to get embedding vector from the last hidden state. Defaults to :data:`"cls"`.
    trust_remote_code                (bool, optional): trust_remote_code for HF datasets or models. Defaults to :data:`False`.
    cache_dir                        (Optional[str], optional): 缓存模型的位置，默认为空
    cobert_dim                       (int, optional): 用于指定 ColBERT 线性层（colbert linear）的维度大小的参数，默认-1隐藏层大小
    batch_size                       (int, optional): 模型在推理或训练过程中，一次性处理的样本数量，默认`256`.
    query_max_length                 (int, optional): Maximum length for query. Defaults to :data:`512`.
    passage_max_length               (int, optional): 指定模型在处理文本段落时的最大长度，默认512
    return_dense                     (bool, optional): If true, will return the dense embedding. Defaults to :data:`True`.
    return_sparse                    (bool, optional): If true, will return the sparce embedding. Defaults to :data:`False`.
    return_colbert_vecs              (bool, optional): If true, will return the colbert vectors. Defaults to :data:`False`.
        
"""
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True)

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1,
                            batch_size=12,
                            max_length=8192,
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]

"""
生成稀疏向量
"""

output_1 = model.encode(sentences_1, return_dense=True, return_sparse=True, return_colbert_vecs=False)
output_2 = model.encode(sentences_2, return_dense=True, return_sparse=True, return_colbert_vecs=False)

# you can see the weight for each token:
print(model.convert_id_to_token(output_1['lexical_weights']))
# [{'What': 0.08356, 'is': 0.0814, 'B': 0.1296, 'GE': 0.252, 'M': 0.1702, '3': 0.2695, '?': 0.04092},
#  {'De': 0.05005, 'fin': 0.1368, 'ation': 0.04498, 'of': 0.0633, 'BM': 0.2515, '25': 0.3335}]


# compute the scores via lexical mathcing
lexical_scores = model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
print(lexical_scores)
# 0.19554901123046875

print(model.compute_lexical_matching_score(output_1['lexical_weights'][0], output_1['lexical_weights'][1]))
# 0.0
