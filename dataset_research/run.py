from dataload_agent import main as run_dataload_agent
from dataprocess_agent import main as run_dataprocess_agent
import argparse
import asyncio
from tqdm import tqdm
import json, time

dataset_names_path = "dataset_research/workspace/dataset_name.jsonl"
jsonl_path = "dataset_research/workspace/jsonl_path.txt"

def read_file(input_file):
    """
    从文本文件中按行读取文件
    参数:input_file: 包含文件路径的文本文件        
    返回:文件内容
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        # 去除每行末尾的换行符，并过滤掉空行
        return [line.strip() for line in f if line.strip()]
    

def extract_from_jsonl(jsonl_file, str:str):
    """
    从 jsonl 文件中提取每行的 str 字段
    参数:jsonl_file (str): jsonl 文件路径
    返回:list: 包含所有 str 的列表
    """
    ids = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())  # 解析单行 JSON
                if str in data:  # 检查是否存在 str 字段
                    ids.append(data[str])
            except json.JSONDecodeError:
                print(f"警告：跳过无效行: {line}")
    return ids

def Dataload_Agent():
    """
    根据数据集名称下载数据集
    
    input: 需要下载的数据集的名称
    output: 下载后数据集的符合文件存储结构的原始数据集
    """
    dataset_names = extract_from_jsonl(dataset_names_path, "dataset_id")
    print(dataset_names)
    for dataset_name in tqdm(dataset_names, desc="数据集下载:"):
        result = asyncio.run(run_dataload_agent(dataset_name=dataset_name))
        print(result)
    
    with open(dataset_names_path, 'w') as f:
        pass  # 清空文件
    return

def Dataprocess_Agent():
    """
    将原始数据的jsonl文件进行格式化处理
    
    """
    
    jsonl_paths = read_file(jsonl_path)
    
    for file in tqdm(jsonl_paths, desc="数据格式处理:"):
        result = asyncio.run(run_dataprocess_agent(raw_file_path=file))
        print(result)
            
    with open(jsonl_path, 'w') as f:
        pass  # 清空文件
    
    return

def execute_RAG(top_k, query):
    query_contents = {"top_k":top_k, "query": query}
    signal_writing_file = "dataset_research/workspace/data_transmit/signal.json"
    data_writing_file = "dataset_research/workspace/data_transmit/data.json"
    with open(signal_writing_file, 'r') as f:
        print("signal_writing_file: ", signal_writing_file)
        signal_contents = json.load(f)
    assert signal_contents['signal'] == 0
    with open(data_writing_file, 'w', encoding='utf-8') as f:
        json.dump(query_contents, f, indent=4, ensure_ascii=False)
    with open(data_writing_file, 'r', encoding='utf-8') as f:
        query_contents_check = json.load(f)
    assert query_contents == query_contents_check
    with open(signal_writing_file, 'w', encoding='utf-8') as f:
        json.dump({'signal': 1}, f, indent=4, ensure_ascii=False)
    print("RAG is running !!!!!!")
    response_finish = False
    while not response_finish:
        with open(signal_writing_file, 'r', encoding='utf-8') as f:
            signal_contents = json.load(f)
        if signal_contents['signal'] == 0:
            response_finish = True
        else:
            time.sleep(10)
    with open(data_writing_file, 'r', encoding='utf-8') as f:
        query_contents = json.load(f)
    
    with open("dataset_research/workspace/dataset_name.jsonl", 'w') as f:
        pass  # 清空文件
    
    for result in query_contents["result"]:
        data = {"dataset_id": result["dataset_id"]}
        with open("dataset_research/workspace/dataset_name.jsonl", "a") as f:
            f.write(json.dumps(data) + "\n")  # 注意末尾加换行符
    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='数据集research')
    parser.add_argument('--top_k', type=int, default=3, help='RAG查询返回的结果数量')
    parser.add_argument('--query', type=str, default='ScienceQA, Multi-modal Multiple Choice')
    args = parser.parse_args()
    
    execute_RAG(top_k=args.top_k, query=args.query)
    
    Dataload_Agent()
    
    Dataprocess_Agent()
    
    