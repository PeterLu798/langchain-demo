# 向量
## 什么是向量  
向量是一种有大小和方向的数学对象。它可以表示为从一个点到另一个点的有向线段。例如，二维空间中的向量可以表示为(x, y)，表示从原点(0, 0)到点(x, y)的有向线段。      
<img src=img/vector.png width=300 />    
**以此类推，我可以用一组坐标(x0,x1,...,xn-1)表示一个N维空间中的向量，N叫做向量的维度。**

## 向量间的相似度计算 
<img src=img/vector_distance.png width=300 />     

### 余弦距离（Cosine Distance）
如上图所示，余弦距离为两个向量之间的夹角度数的**余弦值**。  
当夹角度数越小时，两个向量越相似。当夹角度数趋于0时，余弦值趋于最大值1。**因此余弦距离越大说明两个向量越相似。**
### 欧氏距离（Euclidean Distance）
如上图所示，欧式距离是N维空间中两个点之间的真实距离。   
**所以欧式距离越小说明两个向量越相似。**

## 文本向量化
文本向量化的核心在于将非结构化的文本数据转换为结构化的数值向量。    
常见的文本向量化方法包括独热模型（One Hot Model）、词袋模型（Bag of Words Model）、词频-逆文档频率（TF-IDF）、N元模型（N-Gram）、单词-向量模型（Word2vec）以及文档-向量模型（Doc2vec）等。
### 独热模型（One Hot Model）
#### 原理
1. 问题由来   
在很多机器学习任务中，特征并不总是连续值，而有可能是分类值，比如人的性别、衣服的颜色等。这些称为**离散特征**，离散特征可以分为两类：  
1）离散特征之间没有大小的意义，如衣服颜色等  
2）离散特征之间有大小意义，如衣服尺码X、XL、XXL等  
而我们的目标就是要计算这些特征之间的相似度。对于有大小意义的特征我们假设给他们从小到大映射数值，然后来计算其之间的“距离”（这只是为了方便理解假设的一种方式）。但是对于没有大小意义的离散特征，我们如何计算相似度？
2. one-hot encoding     
one-hot就是为了解决这一问题而来的。one-hot又称一位有效编码，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。   
举个例子说明，比如血型这个特征，一共有四种类别（A,B,AB,O），采用one-hot编码后，会把血型变成有一个4维的稀疏向量：  
A表示为（1,0,0,0）  
B表示为（0,1,0,0）  
AB表示为（0,0,1,0）  
O表示为（0,0,0,1）  
有几个类别，就会生成几维的稀疏向量。有了向量就可以计算它们之间的相似度了。
#### 优缺点
优点：独热编码解决了分类器不好处理属性数据的问题，在一定程度上也起到了扩充特征的作用。它的值只有0和1，不同的类型存储在垂直的空间。   
缺点：类别的数量很多时，特征空间会变得非常大。在这种情况下，一般可以用PCA来减少维度。而且one hot encoding+PCA这种组合在实际中也非常有用。
#### 适用场景
适合：独热编码用来解决类别型数据的离散值问题。   
不适合：将离散型特征进行one-hot编码的作用，是为了让距离计算更合理，但如果特征是离散的，并且不用one-hot编码就可以很合理的计算出距离，那么就没必要进行one-hot编码。
### 词袋模型（Bag of Words Model）
#### 原理

#### 优缺点

#### 适用场景

## 向量数据库
本文重点讲解Milvus
### 安装
Milvus提供了三种部署模式，分别是Milvus Lite、Milvus Standalone、Milvus Distributed。    
Milvus Lite是最轻量化的方式，实质就是一个Python library。这种方式的目的是为了让用户在有限的资源下快速体验Milvus。     
Milvus Standalone是本地单机部署模式，使用Docker来部署。      
Milvus Distributed是生产环境的部署模式，是基于Kubernetes云原生的集群部署。      
下面演示Milvus Standalone部署   
环境：windows   
前提条件：[安装 Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)   
1. 在管理员模式下右击并选择以管理员身份运行，打开 Docker Desktop。 
2. 下载安装脚本并将其保存为standalone.bat 。
```shell
C:\>Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat
```
3. 运行下载的脚本，将 Milvus 作为 Docker 容器启动。
```shell
C:\>standalone.bat start
Wait for Milvus starting...
Start successfully.
To change the default Milvus configuration, edit user.yaml and restart the service.
```
4. 可以使用以下命令管理 Milvus 容器和存储的数据。
```shell
C:\>standalone.bat stop
Stop successfully.

C:\>standalone.bat delete
Delete Milvus container successfully. # Container has been removed.
Delete successfully. # Data has been removed.
```
### 架构特点
#### 密集向量
1. 概念   
密集向量由包含实数的数组组成，其中大部分或所有元素都不为零。与稀疏向量相比，密集向量在同一维度上包含更多信息，因为每个维度都持有有意义的值。这种表示方法能有效捕捉复杂的模式和关系，使数据在高维空间中更容易分析和处理。密集向量通常有固定的维数，从几十到几百甚至上千不等，具体取决于具体的应用和要求。       
1.1 多维表示：每一个点表示一个object，其位置由其维度值决定   
1.2 语义关系：点之间的距离反映了概念之间的语义相似性。距离较近的点表示语义关联度较高的概念      
1.3 聚类效应：相关概念（如Milvus、向量数据库和检索系统）在空间中的位置相互靠近，形成语义聚类     
2. 适用场景     
密集向量主要用于需要理解数据语义的场景，如语义搜索和推荐系统。    
2.1 在语义搜索中，密集向量有助于捕捉查询和文档之间的潜在联系，提高搜索结果的相关性。      
2.2 在推荐系统中，密集矢量有助于识别用户和项目之间的相似性，从而提供更加个性化的建议。
3. 支持的向量模型   
3.1 图像的 CNN 模型（如ResNet、VGG）   
3.2 用于文本的语言模型（如BERT、Word2Vec）
4. 在 Milvus 中使用密集向量    
4.1 添加一个向量字段  
````python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

schema = client.create_schema(
    auto_id=True,
    enable_dynamic_fields=True,
)

schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=4)
````
1）将datatype 设置为受支持的密集向量数据类型   

|类型|描述|
|:----|:----|
|FLOAT_VECTOR|存储 32 位浮点数，常用于表示科学计算和机器学习中的实数。非常适合需要高精度的场景，例如区分相似向量。|
|FLOAT16_VECTOR|存储 16 位半精度浮点数，用于深度学习和 GPU 计算。在精度要求不高的情况下，如推荐系统的低精度召回阶段，它可以节省存储空间。|
|BFLOAT16_VECTOR|存储 16 位脑浮点（bfloat16）数，提供与 Float32 相同的指数范围，但精度有所降低。适用于需要快速处理大量向量的场景，如大规模图像检索。|
2）使用dim 参数指定密集向量的维数    
上面代码中我们添加了一个名为dense_vector 的向量字段来存储密集向量。字段的数据类型为FLOAT_VECTOR ，维数为4 。       

4.2 给向量字段建立索引     
````python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="dense_vector",
    index_name="dense_vector_index",
    index_type="IVF_FLAT",
    metric_type="IP",
    params={"nlist": 128}
)
````
在上面的示例中，使用IVF_FLAT 索引类型为dense_vector 字段创建了名为dense_vector_index 的索引。metric_type 设置为IP ，表示将使用内积作为距离度量。   

#### 二进制向量
1. 概念      
二进制向量是一种将复杂对象（如图像、文本或音频）编码为固定长度二进制值的方法。在 Milvus 中，二进制向量通常表示为比特数组或字节数组。例如，一个 8 维二进制向量可以表示为[1, 0, 1, 1, 0, 0, 1, 0] 。   
二进制向量是一种特殊的数据表示形式，它将传统的高维浮点向量转换为只包含 0 和 1 的二进制向量。这种转换不仅压缩了向量的大小，还降低了存储和计算成本，同时保留了语义信息。
2. 二进制向量特点   
2.1 高效存储：每个维度只需 1 位存储空间，大大减少了存储空间。   
2.2 快速计算：可以使用 XOR 等位运算快速计算向量之间的相似性。    
2.3 固定长度：无论原始文本的长度如何，向量的长度保持不变，从而使索引和检索更加容易。     
2.4 简单直观：直接反映关键词的存在，适合某些专业检索任务。
3. 适用场景    
3.1 对非关键特征的精度要求不高时，二进制向量可以有效保持原始浮点向量的大部分完整性和实用性       
3.2 在计算效率和存储优化至关重要的情况下，例如在搜索引擎或推荐系统等大规模人工智能系统中，**实时处理海量数据**是关键所在     
3.3 通过减小向量的大小，二进制向量有助于降低延迟和计算成本，而不会明显牺牲准确性     
3.4 在移动设备和嵌入式系统等资源受限的环境中也很有用    
3.5 通过使用二进制向量，可以在这些受限环境中实现复杂的人工智能功能，同时保持高性能。   
4. 二进制向量的缺点   
虽然二进制向量在特定场景中表现出色，但其表达能力存在局限性，难以捕捉复杂的语义关系。因此，在实际应用场景中，二进制向量通常与其他向量类型一起使用，以平衡效率和表达能力。
5. 在 Milvus 中使用二进制向量        
5.1 添加二进制向量字段
````python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

schema = client.create_schema(
    auto_id=True,
    enable_dynamic_fields=True,
)

schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
schema.add_field(field_name="binary_vector", datatype=DataType.BINARY_VECTOR, dim=128)
````
在此示例中，添加了一个名为binary_vector 的向量字段，用于存储二进制向量。该字段的数据类型为BINARY_VECTOR ，维数为 128。       

5.2 给向量字段设置索引   
````python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="binary_vector",
    index_name="binary_vector_index",
    index_type="BIN_IVF_FLAT",
    metric_type="HAMMING",
    params={"nlist": 128}
)
````
在上面的示例中，使用BIN_IVF_FLAT 索引类型为binary_vector 字段创建了名为binary_vector_index 的索引。metric_type 设置为HAMMING ，表示使用汉明距离进行相似性测量。

#### 稀疏向量
1. 概念
稀疏向量是高维向量的一种特殊表示形式，其中大部分元素为零，只有少数维度具有非零值   
2. 适用场景    
在涉及需要精确匹配关键词或短语的应用时，稀疏向量往往能提供更精确的结果。    
2.1 文本分析：将文档表示为词袋向量，其中每个维度对应一个单词，只有在文档中出现的单词才有非零值。    
2.2 推荐系统：用户-物品交互矩阵，其中每个维度代表用户对特定物品的评分，大多数用户只与少数物品交互。   
2.3 图像处理：局部特征表示，只关注图像中的关键点，从而产生高维稀疏向量。   
3. 在 Milvus 中使用稀疏向量  
3.1 生成稀疏向量    
稀疏向量可以使用多种方法生成，例如文本处理中的TF-IDF（词频-反向文档频率）和BM25。此外，Milvus 还提供了帮助生成和处理稀疏向量的便捷方法。      
3.2 添加稀疏向量字段
```python
from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

client.drop_collection(collection_name="my_sparse_collection")

schema = client.create_schema(
    auto_id=True,
    enable_dynamic_fields=True,
)

schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
```
1）将datatype 设置为支持的稀疏向量数据类型，即SPARSE_FLOAT_VECTOR  
2）无需指定维度   
3）在此示例中，添加了一个名为sparse_vector 的向量字段，用于存储稀疏向量。该字段的数据类型为SPARSE_FLOAT_VECTOR   
4. 为向量字段设置索引参数

#### 相似度度量类型
相似度量用于衡量向量之间的相似性。选择合适的距离度量有助于显著提高分类和聚类性能。  
目前 Milvus 支持的度量类型如下：   

|度量类型|中文描述|相似性距离值的特征 |相似性距离值范围|
|:----|:----|:----|:----|
|L2|欧氏距离|值越小表示相似度越高。|[0, ∞)|
|IP|内积 |数值越大，表示相似度越高。|[-1, 1]|
|COSINE|余弦相似度|数值越大，表示相似度越高。|[-1, 1]|
|JACCARD|jaccard相似度|数值越小，表示相似度越高。|[0, 1]|
|HAMMING|汉明距离|值越小，表示相似度越高。|0，dim(向量)] [0, dim(vector)|
|BM25|BM25 相似性|根据词频、反转文档频率和文档规范化对相关性进行评分。|[0, ∞)|

不同字段类型与相应度量类型之间的映射关系：  

|字段类型|所属向量|维度范围|支持的度量类型|默认度量类型|
|:----|:----|:----|:----|:----|
|FLOAT_VECTOR|密集向量|2-32,768|COSINE,L2,IP|COSINE|
|FLOAT16_VECTOR|密集向量|2-32,768|COSINE,L2,IP|COSINE|
|BFLOAT16_VECTOR|密集向量|2-32,768|COSINE,L2,IP|COSINE|
|SPARSE_FLOAT_VECTOR|稀疏向量|无需指定维度。|IP,BM25 （仅用于全文检索|IP|
|BINARY_VECTOR|二进制向量|8-32,768*8|HAMMING,JACCARD|HAMMING|

#### 向量字段索引

#### 非向量字段索引



#### 一致性级别










### 使用
#### DataBase
数据库的概念和其他数据库中间件的概念也是一样的，比如Mysql中的DataBase等。
1. 创建数据库
````python
from pymilvus import connections, db
conn = connections.connect(host="127.0.0.1", port=19530)
database = db.create_database("my_database")
````
2. 使用数据库
````python
db.using_database("my_database")
````
或者
````python
conn = connections.connect(
    host="127.0.0.1",
    port="19530",
    db_name="my_database"
)
````
3. 列出数据库
````python
db.list_database()

['default', 'my_database']
````
4. 删除数据库
````python
db.drop_database("my_database")
db.list_database()
['default']
````
#### Collections
Collection 和实体 类似于关系数据库中的表和记录。   
Collection 是一个二维表，具有固定的列和变化的行。每列代表一个字段，每行代表一个实体。  
一个Collection中的要素：
1. Schema 和字段   
在描述一个对象时，我们通常会提到它的属性，如大小、重量和位置。您可以将这些属性用作 Collection 中的字段。每个字段都有各种约束属性，例如向量字段的数据类型和维度。通过创建字段并定义其顺序，可以形成一个 Collections Schema。
2. 主键和 AutoId   
与关系数据库中的主字段类似，Collection 也有一个主字段，用于将实体与其他实体区分开来。主字段中的每个值都是全局唯一的，并与一个特定实体相对应。       
主键可以自己定义，也可以使用Milvus自增主键：   
2.1 定义自增主键   
将datatype 设置为DataType.INT64 ，将is_primary 设置为true，将auto_id 设置为true   
````python
from pymilvus import MilvusClient, DataType

schema = MilvusClient.create_schema()

schema.add_field(
    field_name="my_id",
    datatype=DataType.INT64,
    # highlight-start
    is_primary=True,
    auto_id=True,
    # highlight-end
)
````
3. 索引   
为特定字段创建索引可提高搜索效率。建议您为服务所依赖的所有字段创建索引，其中向量字段的索引是强制性的。      
在 Indexes 中详细说明。
4. 实体
5. 加载和释放   
加载集合是在集合中进行相似性搜索和查询的前提。加载 Collections 时，Milvus 会将所有索引文件和每个字段中的原始数据加载到内存中，以便快速响应搜索和查询。    
搜索和查询是内存密集型操作。为节约成本，建议您释放当前不使用的 Collections。  
6. 搜索和查询   
在 Search & Rerank 详细说明
7. 分区   
7.1 分区是集合的子集，与其父集合共享相同的字段集，每个分区包含一个实体子集。  
7.2 通过将实体分配到不同的分区，可以创建实体组。你可以在特定分区中进行搜索和查询，让 Milvus 忽略其他分区中的实体，提高搜索效率。     
7.3 分区是 Collections 的水平切片。每个分区对应一个数据输入通道。每个 Collections 默认都有一个分区。创建 Collections 时，可以根据预期吞吐量和要插入 Collections 的数据量设置适当的分区数量。
8. 一致性级别   
分布式数据库系统通常使用一致性级别来定义跨数据节点和副本的数据相同性。在创建 Collections 或在 Collections 中进行相似性搜索时，可以分别设置不同的一致性级别。适用的一致性级别有强、有限制的不稳定性、会话和最终。

#### Schema & Data Fields
Schema 用于定义 Collections 的属性和其中的字段。  
1. 创建Field Schema
````python
from pymilvus import DataType, FieldSchema
id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
age_field = FieldSchema(name="age", dtype=DataType.INT64, description="age")
embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128, description="vector")

position_field = FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=256, is_partition_key=True)
````
1.1 name、dtype、description、max_length这些都不必说了    
1.2 dim 向量的维数，密集向量场必须使用，稀疏向量省略  
1.3 default_value 默认字段值    
1.4 is_partition_key 用作 Partition Key 的字段名称
2. 创建Collection schema    
```python
from pymilvus import CollectionSchema
schema = CollectionSchema(fields=[id_field, age_field, embedding_field], auto_id=False, enable_dynamic_field=True, description="desc of a collection")
```
3. 创建Collection
````python
from pymilvus import Collection, connections
conn = connections.connect(host="127.0.0.1", port=19530)
collection_name1 = "tutorial_1"
collection1 = Collection(name=collection_name1, schema=schema, using='default', shards_num=2)
````
#### Insert & Delete
1. Insert
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

data=[
    {"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "color": "pink_8682"},
    {"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "color": "red_7025"},
    {"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "color": "orange_6781"},
]

res = client.insert(
    collection_name="quick_setup",
    data=data
)

print(res)
# Output
# {'insert_count': 10, 'ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
````
2. Upsert    
**upsert操作必须要有主键。**
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

data=[
    {"id": 0, "vector": [-0.619954382375778, 0.4479436794798608, -0.17493894838751745, -0.4248030059917294, -0.8648452746018911], "color": "black_9898"},
    {"id": 1, "vector": [0.4762662251462588, -0.6942502138717026, -0.4490002642657902, -0.628696575798281, 0.9660395877041965], "color": "red_7319"},
    {"id": 2, "vector": [-0.8864122635045097, 0.9260170474445351, 0.801326976181461, 0.6383943392381306, 0.7563037341572827], "color": "white_6465"},
]

res = client.upsert(
    collection_name='quick_setup',
    data=data
)

print(res)
# Output
# {'upsert_count': 10}
````
3. Delete    
可以通过筛选条件或主键删除不再需要的实体。     
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

res = client.delete(
    collection_name="quick_setup",
    # highlight-next-line
    filter="color in ['red_3314', 'purple_7392']"
)

print(res)
# Output
# {'delete_count': 2}
````
通过主键删除实体：
````python
res = client.delete(
    collection_name="quick_setup",
    # highlight-next-line
    ids=[18, 19]
)

print(res)
````
#### Indexes


#### Search & Rerank

#### Partitions


### 高级用法


## 向量模型


# 关键字检索

## 搜索引擎

# LangChain
## 模型 API
把不同的模型，统一封装成一个接口，方便更换模型而不用重构代码。
### OpenAI 模型封装
```shell
pip install --upgrade langchain
pip install --upgrade langchain-openai
pip install --upgrade langchain-community
```
[OpenAI.py](models%2FOpenAI.py)
### Ollama 封装
````shell
pip install -U langchain-ollama
````
[Ollama.py](models%2FOllama.py)
### 多轮对话 Session 封装

## 模型的输入与输出
### Prompt 模板封装
1. PromptTemplate 可以在模板中自定义变量
2. ChatPromptTemplate 用模板表示的对话上下文
3. MessagesPlaceholder 把多轮对话变成模板
### 从文件加载 Prompt 模板
[PromptFromFile.py](io%2FPromptFromFile.py)
### 结构化输出
1. 直接输出 Pydantic 对象
2. 输出指定格式的 JSON
3. 使用 OutputParser 可以按指定格式解析模型的输出，输出形式为Json   
代码示例：[OutputParser.py](io%2Foutput%2FOutputParser.py)
````text
{'year': 2023, 'month': 4, 'day': 6, 'era': 'AD'}
````
4. 使用PydanticOutputParser直接输出对象类型   
代码示例：[PydanticOutputParserDemo.py](io%2Foutput%2FPydanticOutputParserDemo.py)
````text
year=2023 month=4 day=6 era='AD'
````
5. OutputFixingParser 利用大模型做格式自动纠错   
代码示例：[OutputFixingParserDemo.py](io%2Foutput%2FOutputFixingParserDemo.py)

## Function Calling
大模型调用本地方法。

## 数据连接封装
### 文档加载器：Document Loaders
```shell
pip install pymupdf
```
### 文档处理器
#### TextSplitter
```shell
pip install --upgrade langchain-text-splitters
```

### 向量数据库与向量检索
#### 向量模型
1. huggingface
````shell
pip install -qU langchain-huggingface
````
示例：[VectorstoreDemo.py](dataconnection%2FVectorstoreDemo.py)

## 对话历史管理
### 历史记录的剪裁
[TrimMessage.py](history%2FTrimMessage.py)
### 过滤带标识的历史记录
[FilterMessage.py](history%2FFilterMessage.py)

## LCEL
LCEL全称：LangChain Expression Language
LangChain Expression Language（LCEL）是一种声明式语言，可轻松组合不同的调用顺序构成 Chain。LCEL 自创立之初就被设计为能够支持将原型投入生产环境，无需代码更改，从最简单的“提示+LLM”链到最复杂的链（已有用户成功在生产环境中运行包含数百个步骤的 LCEL Chain）。

LCEL 的一些亮点包括：
1. 流支持：使用 LCEL 构建 Chain 时，你可以获得最佳的首个令牌时间（即从输出开始到首批输出生成的时间）。对于某些 Chain，这意味着可以直接从 LLM 流式传输令牌到流输出解析器，从而以与 LLM 提供商输出原始令牌相同的速率获得解析后的、增量的输出。
2. 异步支持：任何使用 LCEL 构建的链条都可以通过同步 API（例如，在 Jupyter 笔记本中进行原型设计时）和异步 API（例如，在 LangServe 服务器中）调用。这使得相同的代码可用于原型设计和生产环境，具有出色的性能，并能够在同一服务器中处理多个并发请求。
3. 优化的并行执行：当你的 LCEL 链条有可以并行执行的步骤时（例如，从多个检索器中获取文档），我们会自动执行，无论是在同步还是异步接口中，以实现最小的延迟。
4. 重试和回退：为 LCEL 链的任何部分配置重试和回退。这是使链在规模上更可靠的绝佳方式。目前我们正在添加重试/回退的流媒体支持，因此你可以在不增加任何延迟成本的情况下获得增加的可靠性。
5. 访问中间结果：对于更复杂的链条，访问在最终输出产生之前的中间步骤的结果通常非常有用。这可以用于让最终用户知道正在发生一些事情，甚至仅用于调试链条。你可以流式传输中间结果，并且在每个 LangServe 服务器上都可用。
6. 输入和输出模式：输入和输出模式为每个 LCEL 链提供了从链的结构推断出的 Pydantic 和 JSONSchema 模式。这可以用于输入和输出的验证，是 LangServe 的一个组成部分。
7. 无缝 LangSmith 跟踪集成：随着链条变得越来越复杂，理解每一步发生了什么变得越来越重要。通过 LCEL，所有步骤都自动记录到 LangSmith，以实现最大的可观察性和可调试性。
8. 无缝 LangServe 部署集成：任何使用 LCEL 创建的链都可以轻松地使用 LangServe 进行部署。

### Pipeline 式调用 PromptTemplate, LLM 和 OutputParser
[ELEL.py](lcel%2FELEL.py)
### 流式输出
[StreamDemo.py](lcel%2FStreamDemo.py)
### 用 LCEL 实现 RAG
[Index.py](rag%2FIndex.py)
### 用 LCEL 实现工厂模式（选）
[ConfigurableDemo.py](lcel%2FConfigurableDemo.py)
### 存储与管理对话历史
[RunnableWithHistory.py](history%2FRunnableWithHistory.py)

## LangServe
LangServe 用于将 Chain 或者 Runnable 部署成一个 REST API 服务。
```shell
# 安装 LangServe
pip install --upgrade langserve[all]

# 也可以只安装一端
pip install "langserve[client]"
pip install "langserve[server]"
```
代码示例：  
[Server.py](lanagservice%2FServer.py)  
[Client.py](lanagservice%2FClient.py)

## 智能体架构：Agent
### 什么是智能体（Agent）
将大语言模型作为一个推理引擎。给定一个任务，智能体自动生成完成任务所需的步骤，执行相应动作（例如选择并调用工具），直到任务完成。

1. 智能体的目的：推理复杂任务
2. 智能体的要素：工具集、短时记忆（内部）、长时记忆（多轮对话历史）

### 智能体类型：ReAct
ReAct智能体架构：
![img.png](img/react.png)
#### LangChain Hub
1. 网站：https://smith.langchain.com/hub
2. LangChain Hub依赖包
```shell
pip install --upgrade langchainhub
```
3. 下载模板
```python
from langchain import hub

# 下载一个现有的 Prompt 模板
react_prompt = hub.pull("hwchase17/react")

print(react_prompt.template)
```
#### google搜索API
```shell
pip install google-search-results
```

### 智能体类型：SelfAskWithSearch
这种模式就是一直向上发问，类似于递归函数，例如问冯小刚老婆演过哪些电影？大模型会一直向上发问：冯小刚老婆是谁？然后搜索她演过的电影。    

### 手动实现一个Agent
#### Agent的核心流程
![agent.png](img/agent.png)
#### 实现步骤
1. 定义结构体Action
2. 实现ReAct逻辑  
2.1 初始化方法


























