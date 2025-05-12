import unittest
import json
import asyncio
import os
from dataset_research.dataprocess_agent import main

class dataprocess_agent_test(unittest.TestCase):
    def setUp(self):
        self.run = main
        self.raw_file_path = "dataset_research/workspace/datasets/leonardPKU_ScienceQA_Test_IMG-default/default.jsonl" # 需要自己手动填写测试的文件路径的名称
        
    def test_successful_run(self):
        result = asyncio.run(self.run(raw_file_path=self.raw_file_path))
        print(result)
        return

if __name__ == "__main__":
    unittest.main() 