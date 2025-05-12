import unittest
import json
import asyncio
import os
from dataset_research.data_description_load import main

class data_description_load_test(unittest.TestCase):
    def setUp(self):
        self.run = main
        
    def test_successful_run(self):
        self.run()
        with open('dataset_research/workspace/data_description/dataset_status.json', 'r') as f:
            """
            测试文件是否存在
            """
            dataset_status = f.read()
        print("文件存在，内容为:",dataset_status)
        # return

if __name__ == "__main__":
    unittest.main()