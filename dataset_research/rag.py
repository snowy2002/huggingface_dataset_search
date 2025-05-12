import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
from opensearchpy import OpenSearch, helpers
import argparse
import os
import uuid
import time
import numpy as np

class HuggingFaceDatasetRAG:
    def __init__(self, index_name="huggingface_datasets", opensearch_host='localhost', opensearch_port=9200, 
                 embedding_model_path=None, use_ssl=True, verify_certs=False,
                 username='admin', password='admin'):
        """
        初始化HuggingFaceDatasetRAG类
        
        参数:
            opensearch_host: OpenSearch主机地址
            opensearch_port: OpenSearch端口
            embedding_model_path: 嵌入模型路径，如果为None则使用默认模型
            use_ssl: 是否使用SSL连接
            verify_certs: 是否验证证书
            username: OpenSearch用户名
            password: OpenSearch密码
        """
        # 初始化OpenSearch客户端
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=False,
            http_auth=(username, password),
            use_ssl=use_ssl,
            verify_certs=verify_certs
        )
        
        # 设置索引名称
        self.index_name = index_name
        
        # 加载嵌入模型
        if embedding_model_path is None:
            embedding_model_path = "intfloat/multilingual-e5-large"
        
        self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.model = AutoModel.from_pretrained(embedding_model_path).to(self.model_device)
        
        # 嵌入维度
        self.embedding_dim = 1024

    def create_index(self, force=False):
        """
        创建OpenSearch索引
        
        参数:
            force: 如果索引已存在，是否强制重新创建
        """
        if self.client.indices.exists(index=self.index_name):
            if force:
                self.client.indices.delete(index=self.index_name)
            else:
                print(f"索引 {self.index_name} 已存在，跳过创建")
                return

        # self.client.indices.create(
        #     index=self.index_name,
        #     body={
        #         "mappings": {
        #             "properties": {
        #                 "id": {"type": "keyword"},
        #                 "dataset_id": {"type": "keyword"},
        #                 "description": {"type": "text"},
        #                 "downloads": {"type": "integer"},
        #                 "tags": {"type": "keyword"},
        #                 "metadata": {"type": "object"},
        #                 "readme": {"type": "text"},
        #                 # 将嵌入向量存储为 float 数组
        #                 "embedding_vector": {
        #                     "type": "object",
        #                     "enabled": False  # 不索引此字段，只存储
        #                 }
        #             }
        #         }
        #     }
        # )
        # 创建索引 - 不使用KNN
        # self.client.indices.create(
        #     index=self.index_name,
        #     body={
        #         "mappings": {
        #             "properties": {
        #                 "id": {"type": "keyword"},
        #                 "dataset_id": {"type": "keyword"},
        #                 "description": {"type": "text"},
        #                 "downloads": {"type": "integer"},
        #                 "tags": {"type": "keyword"},
        #                 "metadata": {"type": "object"},
        #                 "readme": {"type": "text"},
        #                 "embedding_vector": {"type": "dense_vector", "dims": self.embedding_dim}
        #             }
        #         }
        #     }
        # )
        # 创建索引
        print(self.client.info())
        self.client.indices.create(
            index=self.index_name,
            body={
                "settings": {
                    "index.knn": True,
                    # "index.knn.engine": "nmslib",
                    # "index.knn.space_type": "l2"
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "dataset_id": {"type": "text"},
                        "description": {"type": "text"},
                        "downloads": {"type": "integer"},
                        "tags": {"type": "keyword"},
                        # "metadata": {"type": "object"},
                        "readme": {"type": "text"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_dim,
                            "method": {
                                "engine": "lucene",
                                "space_type": "l2",
                                "name": "hnsw",
                                "parameters": {}
                            }
                        }
                    }
                }
            }
        )
        print(f"索引 {self.index_name} 创建成功")

    def average_pool(self, last_hidden_states, attention_mask):
        """
        计算平均池化嵌入
        
        参数:
            last_hidden_states: 模型最后一层隐藏状态
            attention_mask: 注意力掩码
        
        返回:
            平均池化后的嵌入向量
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_embeddings(self, texts):
        """
        获取文本嵌入
        
        参数:
            texts: 文本列表
        
        返回:
            嵌入向量列表
        """
        # 分批处理文本
        embeddings = []
        batch_size = 8  # 可以根据GPU内存调整
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 将文本转换为模型输入
            batch_dict = self.tokenizer(
                batch_texts, 
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(self.model_device)
            
            # 使用模型获取嵌入
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                
            # 计算平均池化嵌入
            batch_embeddings = self.average_pool(
                outputs.last_hidden_state, 
                batch_dict['attention_mask']
            ).cpu().numpy()
            batch_embeddings = batch_embeddings.astype(np.float16)
            # print("batch_embeddings: ", batch_embeddings)
            # print("type(batch_embeddings[0][0]): ", type(batch_embeddings[0][0]))
            embeddings.extend(batch_embeddings)
        
        return embeddings

    def get_embedding(self, text):
        """
        获取单个文本的嵌入
        
        参数:
            text: 文本字符串
        
        返回:
            嵌入向量
        """
        return self.get_embeddings([text])[0]
    
    def index_datasets_by_dir(self, dir_path):
        """
        从目录中索引所有data_description_{num}.jsonl文件到OpenSearch
        
        参数:
            dir_path: 包含data_description_{num}.jsonl文件的目录路径
        """
        print(f"开始从目录 {dir_path} 索引数据集...")
        
        # 检查目录是否存在
        if not os.path.isdir(dir_path):
            raise ValueError(f"目录不存在: {dir_path}")
        
        # 获取目录中所有的data_description_{num}.jsonl文件
        jsonl_files = []
        for f in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, f)) and f.startswith('data_description_') and f.endswith('.jsonl'):
                # 确保文件名格式为data_description_{num}.jsonl
                try:
                    # 尝试提取并解析数字部分
                    filename_parts = f.replace('.jsonl', '').split('_')
                    if len(filename_parts) >= 3 and filename_parts[-1].isdigit():
                        jsonl_files.append(os.path.join(dir_path, f))
                except:
                    continue
        
        if not jsonl_files:
            print(f"警告: 在 {dir_path} 中没有找到data_description_{{num}}.jsonl文件")
            return
        
        print(f"找到 {len(jsonl_files)} 个data_description_{{num}}.jsonl文件")
        
        # 对每个JSONL文件调用index_datasets方法
        for jsonl_file in jsonl_files:
            print(f"处理文件: {os.path.basename(jsonl_file)}")
            self.index_datasets(jsonl_file)
        
        print(f"目录 {dir_path} 中的所有data_description_{{num}}.jsonl文件索引完成")

    def index_datasets(self, jsonl_file, batch_size=500):
        """从JSONL文件索引数据集到OpenSearch，分批处理"""
        print(f"开始从 {jsonl_file} 索引数据集...")
        
        # 读取JSONL文件
        datasets = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dataset = json.loads(line.strip())
                    datasets.append(dataset)
                except json.JSONDecodeError:
                    print(f"无法解析JSON行: {line}")
        
        print(f"读取了 {len(datasets)} 个数据集")
        
        # 分批处理
        for i in range(0, len(datasets), batch_size):
            batch = datasets[i:i+batch_size]
            print(f"处理批次 {i//batch_size + 1}/{(len(datasets)-1)//batch_size + 1}，包含 {len(batch)} 个数据集")
            
            # 准备用于嵌入的文本
            texts = []
            for dataset in batch:
                # 构建包含所有相关信息的文本
                text = f"{dataset.get('ID', '')}\n{dataset.get('Description', '')}"
                
                # 添加标签信息
                tags = dataset.get('tags', [])
                if tags:
                    text += f"\nTags: {', '.join(tags)}"
                    
                # 添加README摘要
                readme = dataset.get('readme', '')
                if readme:
                    # TODO: 如果README太长，只使用前1000个字符
                    readme_summary = readme[:1000] + ("..." if len(readme) > 1000 else "")
                    text += f"\nREADME摘要: {readme_summary}"
                    
                texts.append(text)
            
            # 生成嵌入
            embeddings = self.get_embeddings(texts)
            
            
            # 准备批量索引操作
            actions = []
            for j, (dataset, embedding) in enumerate(zip(batch, embeddings)):
                assert len(embedding.tolist()) == self.embedding_dim
                doc_id = str(uuid.uuid4())
                doc = {
                    "id": doc_id,
                    "dataset_id": dataset.get('ID', ''),
                    "description": dataset.get('Description', ''),
                    "downloads": dataset.get('Downloads', 0),
                    "tags": dataset.get('tags', []),
                    # "metadata": dataset.get('metadata', {}),
                    "readme": dataset.get('readme', ''),
                    "embedding": embedding.tolist()
                }
                
                actions.append({
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": doc
                })
            
            # 执行批量索引
            try:
                helpers.bulk(self.client, actions)
                print(f"成功索引批次 {i//batch_size + 1}")
            except Exception as e:
                print(f"索引批次 {i//batch_size + 1} 时出错: {str(e)}")
                # 如果是磁盘空间问题，暂停一段时间
                if "disk usage exceeded" in str(e):
                    print("磁盘空间不足，暂停30秒...")
                    time.sleep(30)
                
            # 等待一段时间，让系统有时间处理
            time.sleep(2)
        
        print("索引完成")
    # def index_datasets(self, jsonl_file):
    #     """
    #     从JSONL文件索引数据集到OpenSearch
        
    #     参数:
    #         jsonl_file: JSONL文件路径
    #     """
    #     print(f"开始从 {jsonl_file} 索引数据集...")
        
    #     # 读取JSONL文件
    #     datasets = []
    #     with open(jsonl_file, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             try:
    #                 datasets.append(json.loads(line.strip()))
    #             except json.JSONDecodeError:
    #                 print(f"无法解析JSON行: {line}")
        
    #     print(f"读取了 {len(datasets)} 个数据集")
        
    #     # 准备用于嵌入的文本
    #     texts = []
    #     for dataset in datasets:
    #         # 构建包含所有相关信息的文本
    #         text = f"{dataset.get('ID', '')}\n{dataset.get('Description', '')}"
            
    #         # 添加标签信息
    #         tags = dataset.get('tags', [])
    #         if tags:
    #             text += f"\nTags: {', '.join(tags)}"
                
    #         # 添加README摘要
    #         readme = dataset.get('readme', '')
    #         if readme:
    #             # 如果README太长，只使用前1000个字符
    #             readme_summary = readme[:1000] + ("..." if len(readme) > 1000 else "")
    #             text += f"\nREADME摘要: {readme_summary}"
                
    #         texts.append(text)
        
    #     # 获取嵌入
    #     print("生成嵌入...")
    #     embeddings = self.get_embeddings(texts)
        
    #     # 准备批量索引操作
    #     print("准备索引...")
    #     actions = []
    #     for i, (dataset, embedding) in enumerate(zip(datasets, embeddings)):
    #         doc_id = str(uuid.uuid4())
            
    #         # 构建文档
    #         doc = {
    #             "id": doc_id,
    #             "dataset_id": dataset.get('ID', ''),
    #             "description": dataset.get('Description', ''),
    #             "downloads": dataset.get('Downloads', 0),
    #             "tags": dataset.get('tags', []),
    #             "metadata": dataset.get('metadata', {}),
    #             "readme": dataset.get('readme', ''),
    #             "embedding": embedding.tolist()
    #         }
            
    #         # 添加到批量操作
    #         actions.append({
    #             "_index": self.index_name,
    #             "_id": doc_id,
    #             "_source": doc
    #         })
        
    #     # 执行批量索引
    #     print("执行批量索引...")
    #     print("len(actions): ", len(actions))
    #     if actions:
    #         helpers.bulk(self.client, actions)
    #         print(f"成功索引了 {len(actions)} 个数据集")
    #     else:
    #         print("没有数据集需要索引")

    # def search(self, query, top_k=5, search_method="hybrid"):
    #     """
    #     搜索相关数据集
        
    #     参数:
    #         query: 查询字符串
    #         top_k: 返回结果数量
    #         search_method: 搜索方法，可选值: "hybrid"(混合), "semantic"(语义), "keyword"(关键词)
        
    #     返回:
    #         相关数据集列表
    #     """
    #     # 获取查询的嵌入向量
    #     query_embedding = self.get_embedding(query).tolist()
        
    #     if search_method == "semantic":
    #         # 纯语义搜索
    #         body = {
    #             "size": top_k,
    #             "query": {
    #                 "knn": {
    #                     "embedding": {
    #                         "vector": query_embedding,
    #                         "k": top_k
    #                     }
    #                 }
    #             }
    #         }
        
    #     elif search_method == "keyword":
    #         # 纯关键词搜索
    #         body = {
    #             "size": top_k,
    #             "query": {
    #                 "multi_match": {
    #                     "query": query,
    #                     "fields": ["dataset_id^3", "description^2", "readme", "tags^1.5"]
    #                 }
    #             }
    #         }
        
    #     else:  # hybrid
    #         # 混合搜索：语义搜索和关键词搜索的结合
    #         body = {
    #             "size": top_k,
    #             "query": {
    #                 "script_score": {
    #                     "query": {
    #                         "bool": {
    #                             "should": [
    #                                 {
    #                                     "multi_match": {
    #                                         "query": query,
    #                                         "fields": ["dataset_id^3", "description^2", "readme", "tags^1.5"],
    #                                         "boost": 0.4
    #                                     }
    #                                 }
    #                             ]
    #                         }
    #                     },
    #                     "script": {
    #                         "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
    #                         "params": {
    #                             "query_vector": query_embedding
    #                         }
    #                     }
    #                 }
    #             }
    #         }
        
    #     # 执行搜索
    #     response = self.client.search(
    #         index=self.index_name,
    #         body=body
    #     )
        
    #     # 处理结果
    #     results = []
    #     for hit in response['hits']['hits']:
    #         source = hit['_source']
    #         results.append({
    #             "score": hit['_score'],
    #             "dataset_id": source['dataset_id'],
    #             "description": source['description'],
    #             "downloads": source['downloads'],
    #             "tags": source['tags']
    #         })
        
    #     return results

    def check_index_exists(self):
        """
        检查索引是否存在
        """
        try:
            exists = self.client.indices.exists(index=self.index_name)
            if exists:
                print(f"索引 {self.index_name} 存在")
                return True
            else:
                print(f"索引 {self.index_name} 不存在")
                return False
        except Exception as e:
            print(f"检查索引时出错: {str(e)}")
            return False
    
    def search(self, query, top_k=5, search_method="keyword"):
        """
        搜索相关数据集 - 简化版本
        """
        # 首先检查索引是否存在
        if not self.check_index_exists():
            print("索引不存在，无法执行搜索")
            return []

        # 获取查询的嵌入向量
        query_embedding = self.get_embedding(query).tolist()
        assert len(query_embedding) == self.embedding_dim
        # print("query_embedding len: ", len(query_embedding))
        print("search_method: ", search_method)
        if search_method == "semantic":
            # 纯语义搜索
            body = {
                "size": top_k,
                "query": {
                    "match": {
                        "description": query
                    }
                }
            }
        
        elif search_method == "keyword":
            # 纯关键词搜索
            body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["dataset_id^3", "description^2", "readme", "tags^1.5"]
                    }
                }
            }
        
        elif search_method == "hybrid":  # hybrid
            # 混合搜索：语义搜索和关键词搜索的结合
            # body = {
            #     "size": top_k,
            #     "query": {
            #         "script_score": {
            #             # "query": {
            #             #     "bool": {
            #             #         "should": [
            #             #             {
            #             #                 "multi_match": {
            #             #                     "query": query,
            #             #                     "fields": ["dataset_id^3", "description^2", "readme", "tags^1.5"],
            #             #                     "boost": 0.4
            #             #                 }
            #             #             }
            #             #         ]
            #             #     }
            #             # },
            #             "script": {
            #                 "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
            #                 "params": {
            #                     "query_vector": query_embedding
            #                 }
            #             }
            #         }
            #     }
            # }
            # body = {
            #         "size": 10,
            #         "query": {
            #             "script_score": {
            #                 "query": {
            #                     "bool": {
            #                         "must": [
            #                             {
            #                             "match": {
            #                                 "description": query
            #                             }
            #                             }
            #                         ],
            #                         "filter": [
            #                             {
            #                             "exists": {
            #                                 "field": "embedding"
            #                             }
            #                             }
            #                         ]
            #                     }
            #                 },
            #                 "script": {
            #                     "source": """cosineSimilarity(params.query_vector, 'embedding') + 1.0""",
            #                     "params": {
            #                         "query_vector": query_embedding
            #                     }
            #                 }
            #             }
            #         }
            #     }
            body = {
                "size": 10,
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, doc['"'embedding'"']) + 1.0",
                            "params": {
                                "query_vector": query_embedding
                            }
                        }
                    }
                }
            }

            # body =  {
            #     "size" : 2,
            #     "query": {
            #         "knn": {
            #             "embedding": {
            #                 "vector": query_embedding,
            #                 "k": 5
            #             }
            #         }
            #     }
            #     }

        else: # filter
            body = {
            # 筛选出同时包含modality:text和modality:image，或只包含modality:image的数据集
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            # 同时包含modality:text和modality:image
                            "bool": {
                                "must": [
                                    {
                                        "term": {
                                            "tags": "modality:text"
                                        }
                                    },
                                    {
                                        "term": {
                                            "tags": "modality:image"
                                        }
                                    }
                                ]
                            }
                        },
                        # {
                        #     # 只包含modality:image
                        #     "bool": {
                        #         "must": [
                        #             {
                        #                 "term": {
                        #                     "tags": "modality:image"
                        #                 }
                        #             }
                        #         ],
                        #         "must_not": [
                        #             {
                        #                 "term": {
                        #                     "tags": "modality:text"
                        #                 }
                        #             }
                        #         ]
                        #     }
                        # }
                    ]
                }
            }
            }
        
        # 执行搜索
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        
        # 处理结果
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            print(f"hit=======: {hit}")
            results.append({
                "score": hit['_score'],
                "dataset_id": source.get('dataset_id', ''),
                "description": source.get('description', ''),
                "downloads": source.get('downloads', 0),
                "tags": source.get('tags', [])
            })
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Hugging Face数据集RAG系统')
    parser.add_argument('--host', type=str, default='localhost', help='OpenSearch主机地址')
    parser.add_argument('--port', type=int, default=9200, help='OpenSearch端口')
    parser.add_argument('--username', type=str, default='admin', help='OpenSearch用户名')
    parser.add_argument('--password', type=str, default='GAIR-scrl-1', help='OpenSearch密码')
    parser.add_argument('--model', type=str, default='/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/wysi/Dataset-Research/multilingual-e5-large-instruct', help='嵌入模型路径')
    parser.add_argument('--index', action='store_true', help='创建索引并索引数据')
    parser.add_argument('--index_name', type=str, default='huggingface_datasets', help='索引名称')
    parser.add_argument('--force', action='store_true', help='强制重新创建索引')
    parser.add_argument('--index_dir', type=str, default='/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/wysi/Dataset-Research/dataset_research/workspace/data_description', help='要索引的目录路径')
    parser.add_argument('--file', type=str, default='/inspire/hdd/ws-950e6aa1-e29e-4266-bd8a-942fc09bb560/embodied-intelligence/liupengfei-24025/wysi/huggingface_datasets_search/datasets/hugging_face_datasets0.jsonl', help='要索引的JSONL文件路径')
    parser.add_argument('--query', type=str, default='', help='搜索查询')
    parser.add_argument('--top_k', type=int, default=3, help='返回结果数量')
    parser.add_argument('--method', type=str, default='hybrid', 
                        choices=['hybrid', 'semantic', 'keyword', 'filter'], help='搜索方法')
    
    args = parser.parse_args()
    
    # 初始化RAG系统
    rag = HuggingFaceDatasetRAG(
        index_name=args.index_name,
        opensearch_host=args.host,
        opensearch_port=args.port,
        embedding_model_path=args.model,
        use_ssl=True,
        verify_certs=False,
        username=args.username,
        password=args.password
    )
    
    # 创建索引并索引数据
    if args.index:
        rag.create_index(force=args.force)
        if args.index_dir:
            rag.index_datasets_by_dir(args.index_dir)
        elif args.file:
            rag.index_datasets(args.file)
    
    # 执行搜索
    if args.query:
        results = rag.search(args.query, top_k=args.top_k, search_method=args.method)
        print(f"\n查询: {args.query}")
        print(f"搜索方法: {args.method}")
        print(f"找到 {len(results)} 个相关数据集:")
        # TODO: filters可以去metadata搜具体信息，比如只要图片数据集
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['dataset_id']} (得分: {result['score']:.4f})")
            print(f"   下载次数: {result['downloads']}")
            print(f"   标签: {', '.join(result['tags'])}")
            print(f"   描述: {result['description'][:200]}...")

# 示例使用
if __name__ == "__main__":
    main()