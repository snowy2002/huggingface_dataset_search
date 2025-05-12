"""
    下载hugging face上的所有数据的名称以及相关描述，整理为jsonl文件并保存
"""
from collections import defaultdict
import jsonpickle
from huggingface_hub import list_datasets, DatasetCard
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
import requests
import json
import concurrent.futures
from tqdm import tqdm
import time
import threading
import os
import argparse
dataset_path = "dataset_research/workspace/data_description/data_description_"
dataset_status_path = "dataset_research/workspace/data_description/dataset_status.json"
# 用于保护文件写入的锁
file_lock = threading.Lock()
from huggingface_hub import login

hf_token=os.getenv("hf_token")
login(token=hf_token)  # 替换为你的 token

def check_if_dataset_requires_authorization(dataset_name, token=None):
    """判断数据集是否需要授权才能访问"""

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        
    api_url = f"https://datasets-server.huggingface.co/is-valid?dataset={dataset_name}"
    response = requests.get(api_url, headers=headers)
    result = response.json()
    
    # 检查返回结果
    if "error" in result:
        error_msg = result["error"]
        if "gated" in error_msg and "access is not granted" in error_msg:
            return True, "需要授权访问的数据集"
        elif "private" in error_msg:
            return True, "私有数据集"
        else:
            return True, f"其他错误: {error_msg}"
    else:
        return False, "公开数据集，无需授权"


dataset_status = defaultdict(int)  # 默认值为0
def process_dataset(dataset, dataset_path, num):
    """处理单个数据集并保存结果"""
    # 首先检查数据集是否需要授权
    # is_gated, message = check_if_dataset_requires_authorization(dataset.id)
    if dataset.gated:
        dataset_status["数据未授权"] += 1
        return f"跳过 {dataset.id}"
        # return f"跳过 {dataset.id}: {message}"
    
    RETRY_DELAY = 0  # 秒
    MAX_RETRIES = 1 # 最大尝试次数
    attempts = 0
    while attempts < MAX_RETRIES:
        try:
            # 加载 DatasetCard
            card = DatasetCard.load(dataset.id)
            readme_content = card.text[:5000]
            metadata = card.data

            data_to_save = {
                "ID": dataset.id,
                "Description": dataset.description,
                "Downloads": dataset.downloads,
                "tags": dataset.tags,
                "metadata": metadata,
                "readme": readme_content,
            }

            serialized_data = jsonpickle.encode(data_to_save)

            # 使用文件锁写入数据
            with file_lock:
                with open(dataset_path + str(num) + ".jsonl", 'a') as f:
                    f.write(serialized_data + "\n")
            dataset_status["保存成功"] += 1
            return f"✅ {dataset.id} 的元数据已保存"

        except Exception as e:
            attempts += 1
            if attempts < MAX_RETRIES:
                print(f"⚠️ 第 {attempts} 次尝试失败，{dataset.id}，将在 {2 ** (attempts - 1) * RETRY_DELAY} 秒后重试... 错误: {str(e)}")
                if(str(e) == "'DatasetInfo' object has no attribute 'description'"):
                    dataset_status[f"处理时出错 {str(e)}"] += 1
                    return f"❌ 处理 {dataset.id} 时出错（共尝试 {MAX_RETRIES} 次）: {str(e)}"
                time.sleep(2 ** (attempts - 1) * RETRY_DELAY)
            else:
                dataset_status[f"处理时出错 {str(e)}"] += 1
                return f"❌ 处理 {dataset.id} 时出错（共尝试 {MAX_RETRIES} 次）: {str(e)}"

def main(limit = 100):
    # 设置参数
    max_workers = 20  # 线程数量
    # limit = 380000 # 获取的数据集数量
    
    print(f"获取前{limit}条数据集...")
    datasets_list = list(list_datasets(limit=limit))
    # print(len(datasets_list))
    # for dataset_info in datasets_list:
    #     dataset_info_json = jsonpickle.encode(dataset_info)
    #     # print(dataset_info_json)
    #     with open("1.jsonl", 'a') as f:
    #         f.write(dataset_info_json + "\n")
    
    # exit()
    
    # 创建线程池
    print(f"使用 {max_workers} 个线程并行处理数据集...")
    
    batch_size = min(10000, limit)
    num = 0
    for step in range(limit // batch_size):
        # 确保输出文件存在并是空的
        with open(dataset_path + str(num) + ".jsonl", 'w') as f:
            pass  # 创建空文件
        l = step * batch_size
        r = min(l + batch_size, limit)
        # 使用进度条跟踪任务完成情况
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_dataset = {
                executor.submit(process_dataset, dataset, dataset_path, num): dataset 
                for dataset in datasets_list[l : r]
            }
            
            # 使用tqdm创建进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_dataset), 
                            total=len(future_to_dataset), 
                            desc="处理数据集"):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    # 简短地输出结果
                    print(f"\n{result}")
                except Exception as exc:
                    print(f"\n处理 {dataset.id} 时产生异常: {exc}")
        # time.sleep(20)
        num += 1
    export_data = dict(dataset_status)
    # 导出为JSON文件
    with open(dataset_status_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 所有数据下载信息保存至 {dataset_status_path}")

if __name__ == "__main__":
    # 运行主函数
    parser = argparse.ArgumentParser(description='数据描述信息下载')
    parser.add_argument('--size', type=int, default=380000, help='下载数据集描述的数量') # 调试时可选用 100~20000 条数据进行测试, 默认全下载
    args = parser.parse_args()
    main(limit=args.size)