import unittest
import json
import asyncio
import os
from dataset_research.dataload_agent import main

class dataload_agent_test(unittest.TestCase):
    def setUp(self):
        self.run = main
        self.dataset_name = "leonardPKU/ScienceQA_Test_IMG" # 需要自己手动填写测试的数据集的名称
        
    def test_successful_run(self):
        result = asyncio.run(self.run(self.dataset_name))
        print(result)
        return

if __name__ == "__main__":
    unittest.main() 