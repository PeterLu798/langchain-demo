- [向量](#%E5%90%91%E9%87%8F)
  * [什么是向量](#%E4%BB%80%E4%B9%88%E6%98%AF%E5%90%91%E9%87%8F)
  * [向量间的相似度计算](#%E5%90%91%E9%87%8F%E9%97%B4%E7%9A%84%E7%9B%B8%E4%BC%BC%E5%BA%A6%E8%AE%A1%E7%AE%97)
    + [余弦距离（Cosine Distance）](#%E4%BD%99%E5%BC%A6%E8%B7%9D%E7%A6%BBcosine-distance)
    + [欧氏距离（Euclidean Distance）](#%E6%AC%A7%E6%B0%8F%E8%B7%9D%E7%A6%BBeuclidean-distance)
  * [文本向量化](#%E6%96%87%E6%9C%AC%E5%90%91%E9%87%8F%E5%8C%96)
    + [独热模型（One Hot Model）](#%E7%8B%AC%E7%83%AD%E6%A8%A1%E5%9E%8Bone-hot-model)
      - [原理](#%E5%8E%9F%E7%90%86)
      - [优缺点](#%E4%BC%98%E7%BC%BA%E7%82%B9)
      - [适用场景](#%E9%80%82%E7%94%A8%E5%9C%BA%E6%99%AF)
    + [词袋模型（Bag of Words Model）](#%E8%AF%8D%E8%A2%8B%E6%A8%A1%E5%9E%8Bbag-of-words-model)
      - [原理](#%E5%8E%9F%E7%90%86-1)
      - [优缺点](#%E4%BC%98%E7%BC%BA%E7%82%B9-1)
      - [适用场景](#%E9%80%82%E7%94%A8%E5%9C%BA%E6%99%AF-1)
  * [向量数据库](#%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93)
    + [安装](#%E5%AE%89%E8%A3%85)
    + [架构特点](#%E6%9E%B6%E6%9E%84%E7%89%B9%E7%82%B9)
      - [密集向量](#%E5%AF%86%E9%9B%86%E5%90%91%E9%87%8F)
      - [二进制向量](#%E4%BA%8C%E8%BF%9B%E5%88%B6%E5%90%91%E9%87%8F)
      - [稀疏向量](#%E7%A8%80%E7%96%8F%E5%90%91%E9%87%8F)
      - [相似度度量类型](#%E7%9B%B8%E4%BC%BC%E5%BA%A6%E5%BA%A6%E9%87%8F%E7%B1%BB%E5%9E%8B)
      - [向量索引](#%E5%90%91%E9%87%8F%E7%B4%A2%E5%BC%95)
        * [内存索引](#%E5%86%85%E5%AD%98%E7%B4%A2%E5%BC%95)
        * [磁盘索引](#%E7%A3%81%E7%9B%98%E7%B4%A2%E5%BC%95)
        * [GPU索引](#gpu%E7%B4%A2%E5%BC%95)
      - [非向量字段索引](#%E9%9D%9E%E5%90%91%E9%87%8F%E5%AD%97%E6%AE%B5%E7%B4%A2%E5%BC%95)
      - [一致性级别](#%E4%B8%80%E8%87%B4%E6%80%A7%E7%BA%A7%E5%88%AB)
    + [使用](#%E4%BD%BF%E7%94%A8)
      - [DataBase](#database)
      - [Collections](#collections)
      - [Schema & Data Fields](#schema--data-fields)
      - [Insert & Delete](#insert--delete)
      - [创建索引](#%E5%88%9B%E5%BB%BA%E7%B4%A2%E5%BC%95)
      - [Search & Rerank](#search--rerank)
        * [基于ANN算法搜索](#%E5%9F%BA%E4%BA%8Eann%E7%AE%97%E6%B3%95%E6%90%9C%E7%B4%A2)
        * [过滤搜索](#%E8%BF%87%E6%BB%A4%E6%90%9C%E7%B4%A2)
        * [范围搜索](#%E8%8C%83%E5%9B%B4%E6%90%9C%E7%B4%A2)
        * [分组搜索](#%E5%88%86%E7%BB%84%E6%90%9C%E7%B4%A2)
        * [混合搜索](#%E6%B7%B7%E5%90%88%E6%90%9C%E7%B4%A2)
        * [查询](#%E6%9F%A5%E8%AF%A2)
        * [过滤](#%E8%BF%87%E6%BB%A4)
        * [全文检索](#%E5%85%A8%E6%96%87%E6%A3%80%E7%B4%A2)
        * [文本匹配](#%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D)
        * [搜索迭代器](#%E6%90%9C%E7%B4%A2%E8%BF%AD%E4%BB%A3%E5%99%A8)
        * [使用Partition Key](#%E4%BD%BF%E7%94%A8partition-key)
        * [Rerankers](#rerankers)
    + [高级用法](#%E9%AB%98%E7%BA%A7%E7%94%A8%E6%B3%95)
  * [向量模型](#%E5%90%91%E9%87%8F%E6%A8%A1%E5%9E%8B)
- [关键字检索](#%E5%85%B3%E9%94%AE%E5%AD%97%E6%A3%80%E7%B4%A2)
  * [搜索引擎](#%E6%90%9C%E7%B4%A2%E5%BC%95%E6%93%8E)
- [LangChain](#langchain)
  * [模型 API](#%E6%A8%A1%E5%9E%8B-api)
    + [OpenAI 模型封装](#openai-%E6%A8%A1%E5%9E%8B%E5%B0%81%E8%A3%85)
    + [Ollama 封装](#ollama-%E5%B0%81%E8%A3%85)
    + [多轮对话 Session 封装](#%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D-session-%E5%B0%81%E8%A3%85)
  * [模型的输入与输出](#%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%BE%93%E5%85%A5%E4%B8%8E%E8%BE%93%E5%87%BA)
    + [Prompt 模板封装](#prompt-%E6%A8%A1%E6%9D%BF%E5%B0%81%E8%A3%85)
    + [从文件加载 Prompt 模板](#%E4%BB%8E%E6%96%87%E4%BB%B6%E5%8A%A0%E8%BD%BD-prompt-%E6%A8%A1%E6%9D%BF)
    + [结构化输出](#%E7%BB%93%E6%9E%84%E5%8C%96%E8%BE%93%E5%87%BA)
  * [Function Calling](#function-calling)
  * [数据连接封装](#%E6%95%B0%E6%8D%AE%E8%BF%9E%E6%8E%A5%E5%B0%81%E8%A3%85)
    + [文档加载器：Document Loaders](#%E6%96%87%E6%A1%A3%E5%8A%A0%E8%BD%BD%E5%99%A8document-loaders)
    + [文档处理器](#%E6%96%87%E6%A1%A3%E5%A4%84%E7%90%86%E5%99%A8)
      - [TextSplitter](#textsplitter)
    + [向量数据库与向量检索](#%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E4%B8%8E%E5%90%91%E9%87%8F%E6%A3%80%E7%B4%A2)
      - [向量模型](#%E5%90%91%E9%87%8F%E6%A8%A1%E5%9E%8B-1)
  * [对话历史管理](#%E5%AF%B9%E8%AF%9D%E5%8E%86%E5%8F%B2%E7%AE%A1%E7%90%86)
    + [历史记录的剪裁](#%E5%8E%86%E5%8F%B2%E8%AE%B0%E5%BD%95%E7%9A%84%E5%89%AA%E8%A3%81)
    + [过滤带标识的历史记录](#%E8%BF%87%E6%BB%A4%E5%B8%A6%E6%A0%87%E8%AF%86%E7%9A%84%E5%8E%86%E5%8F%B2%E8%AE%B0%E5%BD%95)
  * [LCEL](#lcel)
    + [Pipeline 式调用 PromptTemplate, LLM 和 OutputParser](#pipeline-%E5%BC%8F%E8%B0%83%E7%94%A8-prompttemplate-llm-%E5%92%8C-outputparser)
    + [流式输出](#%E6%B5%81%E5%BC%8F%E8%BE%93%E5%87%BA)
    + [用 LCEL 实现 RAG](#%E7%94%A8-lcel-%E5%AE%9E%E7%8E%B0-rag)
    + [用 LCEL 实现工厂模式（选）](#%E7%94%A8-lcel-%E5%AE%9E%E7%8E%B0%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E9%80%89)
    + [存储与管理对话历史](#%E5%AD%98%E5%82%A8%E4%B8%8E%E7%AE%A1%E7%90%86%E5%AF%B9%E8%AF%9D%E5%8E%86%E5%8F%B2)
  * [LangServe](#langserve)
  * [智能体架构：Agent](#%E6%99%BA%E8%83%BD%E4%BD%93%E6%9E%B6%E6%9E%84agent)
    + [什么是智能体（Agent）](#%E4%BB%80%E4%B9%88%E6%98%AF%E6%99%BA%E8%83%BD%E4%BD%93agent)
    + [智能体类型：ReAct](#%E6%99%BA%E8%83%BD%E4%BD%93%E7%B1%BB%E5%9E%8Breact)
      - [LangChain Hub](#langchain-hub)
      - [google搜索API](#google%E6%90%9C%E7%B4%A2api)
    + [智能体类型：SelfAskWithSearch](#%E6%99%BA%E8%83%BD%E4%BD%93%E7%B1%BB%E5%9E%8Bselfaskwithsearch)
    + [手动实现一个Agent](#%E6%89%8B%E5%8A%A8%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AAagent)
      - [Agent的核心流程](#agent%E7%9A%84%E6%A0%B8%E5%BF%83%E6%B5%81%E7%A8%8B)
      - [实现步骤](#%E5%AE%9E%E7%8E%B0%E6%AD%A5%E9%AA%A4)


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
&emsp;&emsp; 1）将datatype 设置为支持的稀疏向量数据类型，即SPARSE_FLOAT_VECTOR  
&emsp;&emsp; 2）无需指定维度   
&emsp;&emsp; 3）在此示例中，添加了一个名为sparse_vector 的向量字段，用于存储稀疏向量。该字段的数据类型为SPARSE_FLOAT_VECTOR   

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

#### 向量索引
##### 内存索引
Milvus支持的大多数向量索引都使用ANNS（近似近邻检索）算法。ANNS 的核心理念不再局限于返回最精确的结果，而是只搜索目标的近邻。ANNS 通过在可接受的范围内牺牲精确度来提高检索效率。   
ANNS 向量索引可分为四种类型：
* Tree-based
* Graph-based
* Hash-based
* Quantization-based  

```text
Tips: 
精确率=检索出的相关信息量 / 检索出的信息总量
召回率=检索出的相关信息量 / 系统中的相关信息总量
```

1、密集向量支持的索引类型

|支持的索引|分类| 场景                             |
|:----|:----|:-------------------------------|
|FLAT|N/A| 1）数据集相对较小 2）需要 100% 的召回率       |
|IVF_FLAT|N/A| 1）高速查询  2）要求尽可能高的召回率           |
|IVF_SQ8|Quantization-based| 1）极高速查询 2）内存资源有限 3）可接受召回率略有下降  |
|IVF_PQ|Quantization-based| 1）高速查询 2）内存资源有限 3）可略微降低召回率     |
|HNSW|Graph-based| 1）极高速查询 2）要求尽可能高的召回率 3）内存资源大   |
|HNSW_SQ|Quantization-based| 1）非常高速的查询 2）内存资源有限 3）可略微降低召回率  |
|HNSW_PQ|Quantization-based| 1）中速查询 2）内存资源非常有限 3）在召回率方面略有妥协 |
|HNSW_PRQ|Quantization-based| 1）中速查询 2）内存资源非常有限 3）召回率略有下降    |
|SCANN|Quantization-based| 1）极高速查询 2）要求尽可能高的召回率 3）内存资源大   |

2、二进制向量支持的索引类型

|支持的索引|分类|场景|
|:----|:----|:----|
|BIN_FLAT|Quantization-based|1）取决于相对较小的数据集。2）要求完全准确。3）无需压缩。4）保证精确的搜索结果。|
|BIN_IVF_FLAT|Quantization-based|1）高速查询 2）要求尽可能高的召回率|

3、稀疏向量支持的索引类型

|支持的索引| 分类            |场景|
|:----|:--------------|:----|
|SPARSE_INVERTED_INDEX	| Inverted index|1）取决于相对较小的数据集。2）要求 100%的召回率。|

##### 磁盘索引
Milvus磁盘索引基于DiskANN算法。   
1. DiskANN 默认为禁用。如果你更喜欢内存索引而不是磁盘索引，建议你禁用该功能以获得更好的性能。  
2. 要重新启用该功能，可将queryNode.enableDisk 设为true  
3. Milvus 实例在 Ubuntu 18.04.6 或更高版本上运行。 
4. Milvus 数据路径应挂载到 NVMe SSD 上，以充分发挥性能。

##### GPU索引
Milvus 支持各种 GPU 索引类型，以加快搜索性能和效率，尤其是在高吞吐量和高调用场景中。   
值得注意的是，与使用 CPU 索引相比，使用 GPU 索引并不一定能减少延迟。**如果想完全最大化吞吐量，则需要极高的请求压力或大量的查询向量。**    
Milvus 目前支持的 GPU 索引类型如下表：  

|索引类型|GPU内存要求|场景|
|:----|:----|:----|
|GPU_CAGRA|内存使用量约为原始向量数据的 1.8 倍。||
|GPU_IVF_FLAT|需要与原始数据大小相等的内存。||
|GPU_IVF_PQ|占用内存较少，具体取决于压缩参数设置。||
|GPU_BRUTE_FORCE|需要与原始数据大小相等的内存。|GPU_BRUTE_FORCE 专为对召回率要求极高的情况定制，通过将每个查询与数据集中的所有向量进行比较，保证召回率为 1。|

#### 非向量字段索引
Milvus 支持向量字段和非向量字段的联合过滤搜索。为了提高非向量字段的搜索效率，Milvus 从 2.1.0 版开始引入了标量字段索引。    
1. Milvus中标量字段索引原理   
1.1 用逻辑操作符先将标量字段组织成布尔表达式   
1.2 当 Milvus 收到带有这种布尔表达式的搜索请求时，它会将布尔表达式解析为抽象语法树（AST），以生成用于属性筛选的物理计划    
1.3 然后，Milvus 在每个分段中应用物理计划，生成一个比特集作为过滤结果，并将结果作为向量搜索参数，以缩小搜索范围    
1.4 在这种情况下，向量搜索的速度在很大程度上依赖于属性过滤的速度。    
2. 标量字段索引算法  
Milvus 的标量字段索引算法旨在实现低内存占用率、高过滤效率和短加载时间。这些算法主要分为两类：auto indexing（自动索引）和inverted indexing（反转索引）     
2.1 auto indexing       
Milvus 提供了AUTOINDEX 选项，让你无需手动选择索引类型。调用create_index 方法时，如果没有指定index_type ，Milvus 会根据数据类型自动选择最合适的索引类型。       
2.2 inverted indexing  
倒排索引有两个主要部分：术语字典和倒排列表。如下图例子：  
<img src=img/scalar_index_inverted.png width=600 />     
图中有两行数据，假定它们的所在列的主键分别是0和1，术语词典则记录了所有标记词（按字母顺序排列）。   
1）点查询：例如，在搜索包含单词Milvus 的文档时，首先要检查术语字典中是否存在Milvus。如果没有找到，则没有文档包含该词。但如果找到了，则会检索与Milvus相关的倒序列表，指出包含该词的文档。这种方法比在一百万个文档中进行暴力搜索要有效得多，因为排序后的术语词典大大降低了查找Milvus 这个词的时间复杂度。   
2）范围查询：范围查询（如查找单词字母大于very 的文档）的效率也能通过排序术语字典得到提高。这种方法比暴力搜索更有效，能提供更快、更准确的结果。

#### 一致性级别
Milvus支持的一致性级别包含：  
* Strong（强）  
* Bounded（有限制的）  
* Eventually（最终一致性）  
* Session（会话）  

默认为Bounded。    

设置一致性级别：
````python
client.create_collection(
    collection_name="my_collection",
    schema=schema,
    # highlight-next
    consistency_level="Strong",
)
````
consistency_level 参数的可能值是Strong 、Bounded 、Eventually 和Session。

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
在 创建索引 中详细说明。
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
#### 创建索引
这里重点讲解**基于内存索引**如何创建、删除等。

|数据类型| 度量类型                               | 索引类型                                                                                          |
|:----|:-----------------------------------|:----------------------------------------------------------------------------------------------|
|密集向量| 欧氏距离 (L2)<br/>内积 (IP)<br/>余弦相似度 (COSINE)| FLAT<br/>IVF_FLAT<br/>IVF_SQ8<br/>IVF_PQ<br/>GPU_IVF_FLAT<br/>GPU_IVF_PQ<br/>HNSW<br/>DISKANN |
|二进制向量|Jaccard (JACCARD)<br/>汉明 (HAMMING)| BIN_FLAT<br/>BIN_IVF_FLAT                                                                     |
|稀疏向量|内积 (IP)| SPARSE_INVERTED_INDEX                                                                         |
|标量字段|N/A| auto index（默认为空即为auto）<br/>INVERTED<br/>BITMAP                                                |

1、创建密集向量索引
````python
index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector_field",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

client.create_index(
    collection_name="customized_setup",
    index_params=index_params,
    sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
)
````
IVF_FLAT详解：
1. 适用的场景是一、高速查询 二、要求尽可能高的召回率
2. IVF_FLAT 将向量数据划分为nlist 个聚类单元，然后比较目标输入向量与每个聚类中心之间的距离。根据系统设置查询的簇数 (nprobe)，相似性搜索结果仅根据目标输入与最相似簇中向量的比较结果返回--大大缩短了查询时间。
3. 通过调整nprobe ，可以在特定情况下找到准确性和速度之间的理想平衡。      
4. 索引构建参数  

|参数|说明|范围|默认值|
|:----|:----|:----|:----|
|nlist|群组单位数|[1, 65536]|128|

5. 搜索参数    
普通搜索:  

|参数|说明|范围|默认值|
|:----|:----|:----|:----|
|nprobe|要查询的单位数|[1，nlist］|8|

&emsp;&emsp; 范围搜索: 

|参数|说明|范围|默认值|
|:----|:----|:----|:----|
|max_empty_result_buckets|未返回任何搜索结果的桶的最大数量。<br/>这是一个范围搜索参数，当连续空桶的数量达到指定值时，将终止搜索过程。<br/>增加该值可以提高召回率，但代价是增加搜索时间。|[1, 65535]|2|

2、创建二进制向量索引    
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
在上面的示例中：   
索引名为：binary_vector_index   
索引类型为：BIN_IVF_FLAT          
度量类型为：HAMMING，表示使用汉明距离进行相似性测量。   

BIN_IVF_FLAT详解：  
该指标与 IVF_FLAT 完全相同，只是只能用于二进制嵌入。   
因此其原理、参数都可参考IVF_FLAT的。

3、创建稀疏向量索引   
````python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="sparse_vector",
    index_name="sparse_inverted_index",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="IP",
    params={"inverted_index_algo": "DAAT_MAXSCORE"},
)
````
1）index_type为索引类型，有效值： 
*   SPARSE_INVERTED_INDEX:稀疏向量的通用反转索引。    

2）metric_type:用于计算稀疏向量之间相似性的度量。有效值：   
*   IP (内积）：使用点积来衡量相似性。
*   BM25:通常用于全文搜索，侧重于文本相似性。

3）params.inverted_index_algo:用于建立和查询索引的算法。有效值：
*   "DAAT_MAXSCORE" (默认值）：maxScore 通过跳过可能影响最小的术语和文档，为高 k 值或包含大量术语的查询提供更好的性能。
*   "DAAT_WAND"：wand算法利用最大影响分数跳过非竞争性文档，从而评估较少的命中文档，但每次命中的开销较高。
*   "TAAT_NAIVE"

SPARSE_INVERTED_INDEX详解：  
1. 索引构建参数

|参数|取值|含义|
|:----|:----|:----|
|inverted_index_algo|DAAT_MAXSCORE||
||DAAT_WAND||
||TAAT_NAIVE||

2. 搜索参数   

|参数|含义|
|:----|:----|
|drop_ratio_search|允许在搜索过程中对查询向量中的小值进行微调。例如，使用{"drop_ratio_search": 0.2} 时，查询向量中最小的 20% 值将在搜索过程中被忽略。|

#### Search & Rerank
##### 基于ANN算法搜索
1、简单介绍原理  
常用的向量相似性检索算法有两种：  
一、KNN(k-Nearest Neighbors) 翻译为K-近邻算法。此算法必须将向量空间中的所有向量与搜索请求中携带的查询向量进行比较，然后找出最相似的向量，这既耗时又耗费资源。     
二、ANN(Approximate Nearest Neighbor) 翻译为近似近邻。该算法要求提供一个索引文件，记录向量 Embeddings 的排序顺序。当收到搜索请求时，可以使用索引文件作为参考，快速找到可能包含与查询向量最相似的向量嵌入的子组。然后，你可以使用指定的度量类型来测量查询向量与子组中的向量之间的相似度，根据与查询向量的相似度对组成员进行排序，并找出前 K 个组成员。       

2、单个向量检索   
在 ANN 搜索中，单向量搜索指的是只涉及一个查询向量的搜索。根据预建索引和搜索请求中携带的度量类型，Milvus 将找到与查询向量最相似的前 K 个向量。   
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
res = client.search(
    collection_name="my_collection",
    anns_field="vector",
    data=[query_vector],
    limit=3, # 返回3条数据
    search_params={"metric_type": "IP"} # 度量类型为"IP"
)

for hits in res:
    for hit in hits:
        print(hit)
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ]
# ]

````
3、多个向量检索   
````python
query_vectors = [
    [0.041732933, 0.013779674, -0.027564144, -0.013061441, 0.009748648],
    [0.0039737443, 0.003020432, -0.0006188639, 0.03913546, -0.00089768134]
]

res = client.search(
    collection_name="my_collection",
    data=query_vectors,
    limit=3,
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
# Output
#
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ],
#     [
#         {
#             "id": 730,
#             "distance": 0.04431751370429993,
#             "entity": {}
#         },
#         {
#             "id": 333,
#             "distance": 0.04231833666563034,
#             "entity": {}
#         },
#         {
#             "id": 232,
#             "distance": 0.04221535101532936,
#             "entity": {}
#         }
#     ]
# ]
````

4、在分区中进行 ANN 搜索
````python
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]
res = client.search(
    collection_name="my_collection",
    # highlight-next-line
    partition_names=["partitionA"], # 指定要查询的分区
    data=[query_vector],
    limit=3,
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {}
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {}
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {}
#         }
#     ]
# ]
````

5、指定输出实体字段  
````python
query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592],

res = client.search(
    collection_name="quick_setup",
    data=[query_vector],
    limit=3, # The number of results to return
    search_params={"metric_type": "IP"},
    # highlight-next-line
    output_fields=["color"] # 这里指定输出color字段
)

print(res)
# [
#     [
#         {
#             "id": 551,
#             "distance": 0.08821295201778412,
#             "entity": {
#                 "color": "orange_6781"
#             }
#         },
#         {
#             "id": 296,
#             "distance": 0.0800950899720192,
#             "entity": {
#                 "color": "red_4794"
#             }
#         },
#         {
#             "id": 43,
#             "distance": 0.07794742286205292,
#             "entity": {
#                 "color": "grey_8510"
#             }
#         }
#     ]
# ]
````

6、增强 ANN 检索     
下面这些搜索都是为了增强ANN检索。

##### 过滤搜索
ANN 搜索能找到与指定向量嵌入最相似的向量嵌入。但是，搜索结果不一定总是正确的。您可以在搜索请求中包含过滤条件，这样 Milvus 就会在进行 ANN 搜索前进行元数据过滤，将搜索范围从整个 Collections 缩小到只搜索符合指定过滤条件的实体。   
在 Milvus 中，过滤搜索根据应用过滤的阶段分为两种类型——**标准过滤**和**迭代过滤**。    
1、标准过滤     
其原理总结如下：   
a. 首先根据过滤条件筛选下符合条件的实体     
b. 在过滤后的实体中进行 ANN 搜索   
c. 返回前 K 个实体。   
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]

res = client.search(
    collection_name="my_collection",
    data=[query_vector],
    limit=5,
    # highlight-start
    filter='color like "red%" and likes > 50', # 这里使用标准过滤
    output_fields=["color", "likes"]
    # highlight-end
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
````

2、迭代过滤  
2.1 标准过滤的缺点   
标准过滤过程能有效地将搜索范围缩小到很小的范围。但是，**过于复杂的过滤表达式**可能会导致非常高的搜索延迟。在这种情况下，迭代过滤可以作为一种替代方法，帮助减少标量过滤的工作量。     
2.2 迭代过滤的原理  
使用迭代过滤的搜索以迭代的方式执行向量搜索。迭代器返回的每个实体都要经过标量过滤，这个过程一直持续到达到指定的 topK 结果为止。   
2.3 迭代过滤的缺点   
不过，值得注意的是，迭代器一次处理一个实体。这种顺序方法可能会导致较长的处理时间或潜在的性能问题，尤其是在对大量实体进行标量过滤时。   
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

query_vector = [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]

res = client.search(
    collection_name="my_collection",
    data=[query_vector],
    limit=5,
    # highlight-start
    filter='color like "red%" and likes > 50',
    output_fields=["color", "likes"],
    search_params={
        "hints": "iterative_filter"
    }    
    # highlight-end
)

for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)
````
##### 范围搜索
<img src=img/range-search.png width=600 />        

用上面这个图来解释什么是范围搜索。    
图中的查询条件为：
````python
data = [0.1,-0.2,0.3,-0.4,0.5],
limit = 3,
search_params = {
  "metric_type":"COSINE",
  "params": {
    "radius": 0.4,
    "range_filter":0.6
  }
}
````
代码中使用 COSINE 距离，params的意思是搜索相似度在 **(radius,range_filter]** 范围内的向量，即 (0.4,0.6] 范围内的向量。   
这就是范围搜索。    
注意，对于不同的度量类型，radius 和 range_filter 的取值也不同，具体看下表：

|度量类型|名称|设置 radius 和 range_filter 的要求|
|:----|:----|:----|
|L2|L2 距离越小，表示相似度越高。|要忽略最相似的向量 Embeddings，请确保<br/>range_filter <= 距离 <radius|
|IP|IP 距离越大，表示相似度越高。|要忽略最相似的向量嵌入，请确保<br/>radius < 距离 <=range_filter|
|COSINE|COSINE 距离越大，表示相似度越高。|要忽略最相似的向量嵌入，请确保<br/>radius < 距离 <=range_filter|
|JACCARD|Jaccard 距离越小，表示相似度越高。|要忽略最相似的向量嵌入，请确保<br/>range_filter <= 距离 <radius|
|HAMMING|汉明距离越小，表示相似度越高。|要忽略最相似的向量嵌入，请确保<br/>range_filter <= 距离 <radius|

##### 混合搜索
密集向量和稀疏向量上面已经做了详细的说明。混合搜索就是同时进行多个 ANN 搜索、对这些 ANN 搜索的多组结果进行 Rerankers 并最终返回一组结果的搜索方法。   
混合搜索的工作流程如下：   
1、通过BERT和Transformers 等向量模型生成密集向量  
2、通过BM25、BGE-M3、SPLADE 等向量模型生成稀疏向量  
3、创建 Collections 并定义 Collections Schema，其中包括密集向量场和稀疏向量   
4、将稀疏密集向量插入上一步刚刚创建的 Collections 中   
5、进行混合搜索：稠密向量上的 ANN 搜索将返回一组前 K 个最相似的结果，稀疏向量上的文本匹配也将返回一组前 K 个结果    
6、归一化：对两组 K 强结果的得分进行归一化，将得分转换为 [0,1] 之间的范围   
7、选择适当的 Rerankers 策略，对两组 Top-K 结果进行合并和重排，最终返回一组 Top-K 结果    

一个混合搜索的示例：[示例](https://milvus.io/docs/zh/multi-vector-search.md#Examples)
##### 查询
查询是针对元数据进行过滤，本节是对 **过滤搜索** 一节的补充。    
1、直接使用主键进行get查询
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

res = client.get(
    collection_name="query_collection",
    ids=[0, 1, 2],
    output_fields=["vector", "color"]
)

print(res)
````

2、使用filter进行过滤查询  
````python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

res = client.query(
    collection_name="query_collection",
    filter="color like \"red%\"",
    output_fields=["vector", "color"],
    limit=3
)
````

3、使用查询迭代器   
````python
from pymilvus import connections, Collection

connections.connect(
    uri="http://localhost:19530",
    token="root:Milvus"
)

collection = Collection("query_collection")

iterator = collection.query_iterator(
    batch_size=10,
    expr="color like \"red%\"",
    output_fields=["color"]
)

results = []

while True:
    result = iterator.next()
    if not result:
        iterator.close()
        break

    print(result)
    results += result
````

##### 过滤
本节也是对 **过滤搜索** 一节的补充。   
1、Milvus支持的比较操作符
* == 等于
* != 不等于
* &gt; 大于
* &lt; 小于
* &gt;= 大于或等于
* &lt;= 小于或等于

2、Milvus支持的范围操作符
* in : 用于匹配特定集合或范围内的值。
* like : 用于匹配模式（主要用于文本字段）
````python
filter = 'name LIKE "Prod%"' # 查找name 以 "Prod "开头的所有产品
filter = 'name LIKE "%XYZ"' # 查找name 以 "XYZ "结尾的所有产品
filter = 'name LIKE "%Pro%"' # 查找name 中包含 "Pro "一词的所有产品
````

##### 全文检索



##### 文本匹配

##### 搜索迭代器


##### Rerankers

### 高级用法


## 向量模型


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


























