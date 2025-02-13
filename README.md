# 向量
## 什么是向量  
向量是一种有大小和方向的数学对象。它可以表示为从一个点到另一个点的有向线段。例如，二维空间中的向量可以表示为(x, y)，表示从原点(0, 0)到点(x, y)的有向线段。     
![vector.png](img/vector.png)       
**以此类推，我可以用一组坐标(x0,x1,...,xn-1)表示一个N维空间中的向量，N叫做向量的维度。**
## 向量间的相似度计算
![vector_distance.png](img/vector_distance.png)     
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
### 

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


























