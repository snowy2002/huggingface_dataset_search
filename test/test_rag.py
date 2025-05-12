import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from dataset_research.rag import HuggingFaceDatasetRAG
import argparse

class TestRAGSystem(unittest.TestCase):
    
    def setUp(self):
        """测试前的设置"""
        parser = argparse.ArgumentParser(description='Hugging Face数据集RAG系统')
        parser.add_argument('--host', type=str, default='localhost', help='OpenSearch主机地址')
        parser.add_argument('--port', type=int, default=9200, help='OpenSearch端口')
        parser.add_argument('--username', type=str, default='admin', help='OpenSearch用户名')
        parser.add_argument('--password', type=str, default='GAIR-scrl-1', help='OpenSearch密码')
        parser.add_argument('--model', type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/public/yxzheng/ckpts/multilingual-e5-large-instruct', help='嵌入模型路径')
        parser.add_argument('--index', action='store_true', help='创建索引并索引数据')
        parser.add_argument('--index_name', type=str, default='huggingface_datasets', help='索引名称')
        parser.add_argument('--force', action='store_true', help='强制重新创建索引')
        parser.add_argument('--index_dir', type=str, default='dataset_research/workspace/data_description', help='要索引的目录路径')
        parser.add_argument('--file', type=str, default='dataset_research/workspace/data_description/data_description_0.jsonl', help='要索引的JSONL文件路径')
        parser.add_argument('--query', type=str, default='English dataset', help='搜索查询')
        parser.add_argument('--top_k', type=int, default=10, help='返回结果数量')
        parser.add_argument('--method', type=str, default='semantic', 
                            choices=['hybrid', 'semantic', 'keyword', 'filter'], help='搜索方法')
        args = parser.parse_args()
        self.args = args
        
        # 初始化RAG系统
        self.rag_system = HuggingFaceDatasetRAG(
            index_name=args.index_name,
            opensearch_host=args.host,
            opensearch_port=args.port,
            embedding_model_path=args.model,
            use_ssl=True,
            verify_certs=False,
            username=args.username,
            password=args.password
        )
    
    def test_initialization(self):
        """测试 RAG 系统初始化"""
        self.assertIsNotNone(self.rag_system)
    
    def test_answer_generation(self):
        """测试答案生成功能"""
        results = self.rag_system.search(self.args.query, top_k=self.args.top_k, search_method=self.args.method)
        print(f"\n查询: {self.args.query}")
        print(f"搜索方法: {self.args.method}")
        print(f"找到 {len(results)} 个相关数据集:")
        # TODO: filters可以去metadata搜具体信息，比如只要图片数据集
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['dataset_id']} (得分: {result['score']:.4f})")
            print(f"   下载次数: {result['downloads']}")
            print(f"   标签: {', '.join(result['tags'])}")
            print(f"   描述: {result['description'][:200]}...")
        self.assertIsNotNone(results)
        self.assertEqual(len(results), self.args.top_k)
        self.assertEqual(results[0]['dataset_id'], 'CGIAR/TranslationDataset_AgriQueries')
        self.assertEqual(results[0]['downloads'], 5)


if __name__ == '__main__':
    unittest.main()
