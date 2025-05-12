# Dataset-Research

![image](https://github.com/user-attachments/assets/af0f44de-f085-4098-a08b-94789d862875)

## 环境安装
要开始使用这个仓库，您需要安装所需的依赖项。您可以通过运行以下命令来完成此操作：     

```bash
git clone https://github.com/GAIR-NLP/Dataset-Research.git
cd Dataset-Research

# 创建虚拟环境 opensearch 用来提供 RAG 服务
conda create -n opensearch python=3.10
conda activate opensearch
pip install -r opensearch_resuirements.txt

# 创建虚拟环境 dataset_research 用来执行 Agent
conda create -n dataset_research python=3.10
conda activate dataset_research
pip install -r resuirements.txt
```

## 配置环境变量
打开 `.env` 文件，配置 `api_key`，`base_url`以及`hf_token`
配置好后执行
```bash
conda activate dataset_research
. script/env.sh
echo $api_key
```
如果显示为你配置的`api_key`，则说明配置成功。

## 下载数据集描述信息
### 下载huggingface所有的数据集描述信息
可以先选用`--size 1000`进行测试，然后选取`--size 380000`将所有数据描述信息进行下载。
```bash
conda activate dataset_research # 切换到 dataset_research 环境中
python dataset_research/data_description_load.py --size [1000, 380000]
```
数据集描述信息下载完成后会存放到 `dataset_research/workspace/data_description` 文件夹中

## OpenSearch使用
### 1. 安装OpenSearch（详细信息可以参考[Installation Guideline](https://docs.opensearch.org/docs/latest/install-and-configure/install-opensearch/debian/)）
  - 在A800服务器已经安装并启动，可以直接使用
  - 运行下面代码，确认OpenSearch是否已经启动
    ```bash
    curl -X GET https://localhost:9201 -u 'admin:H3Cd?0Pu6R' --insecure
    ```

  - 返回下列信息说明OpenSearch已经启动
    ```bash
    {
      "name" : "a800",
      "cluster_name" : "opensearch",
      "cluster_uuid" : "i1xFMhksSuujE2zXlPNtfA",
      "version" : {
        "distribution" : "opensearch",
        "number" : "3.0.0",
        "build_type" : "deb",
        "build_hash" : "dc4efa821904cc2d7ea7ef61c0f577d3fc0d8be9",
        "build_date" : "2025-05-03T06:23:34.992456558Z",
        "build_snapshot" : false,
        "lucene_version" : "10.1.0",
        "minimum_wire_compatibility_version" : "2.19.0",
        "minimum_index_compatibility_version" : "2.0.0"
      },
      "tagline" : "The OpenSearch Project: https://opensearch.org/"
    }
    ```
### 2. 插入表
建立索引表，并且将下载好的数据集描述信息以embedding的形式插入表中，作为RAG数据库
```bash
conda activate opensearch
python ./dataset_research/rag.py --host localhost --port 9201 --username admin \ 
--password H3Cd?0Pu6R --model /data2/ckpts/multilingual-e5-large-instruct \ 
--index --index_name {索引名称} --index_dir {数据集描述信息存放的文件路径}
```

例如：
```bash
python ./dataset_research/rag.py --host localhost --port 9201 --username admin \ 
--password H3Cd?0Pu6R --model /data2/ckpts/multilingual-e5-large-instruct --index \ 
--index_name 'huggingface'  --index_dir 'dataset_research/workspace/data_description'
```


### 3. 查询表中内容
在RAG数据库查询相关内容
```bash
python ./dataset_research/rag.py --host localhost --port 9201 --username admin \ 
--password H3Cd?0Pu6R --model /data2/ckpts/multilingual-e5-large-instruct \ 
--index_name {索引名称} --query {查询请求} --top_k {返回结果数量} \ 
--method {搜索方法:['hybrid', 'semantic', 'keyword', 'filter']}
```

例如：
```bash
python ./dataset_research/rag.py --host localhost --port 9201 --username admin \  
--password H3Cd?0Pu6R --model /data2/ckpts/multilingual-e5-large-instruct \
--index_name 'huggingface'  --query 'english' --top_k 3 --method filter
```

 

## 运行Agent
### 1. 启动 rag_server 服务程序
在此之前，你需要确保已经建立数据索引，并且在表中插入数据
```bash
conda activate opensearch
python ./dataset_research/rag_server.py --host localhost --port 9201 --username admin \ 
--password H3Cd?0Pu6R --model /data2/ckpts/multilingual-e5-large-instruct \  
--index_name 'huggingface'
```


### 2. 运行 Agent
原来的终端界面作为服务端，新建一个终端界面，并执行
```bash
conda activate dataset_research
python dataset_research/run.py
```
