import json
import os
from typing import List, Dict, Any, Optional
from agents import function_tool, RunContextWrapper, Agent, Runner
import asyncio
from openai import AsyncOpenAI
from agents import (
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)
import sys
import subprocess
jsonl_path = "dataset_research/workspace/jsonl_path.txt"

api_key=os.getenv("api_key")
base_url=os.getenv("base_url")

client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=500,
)

set_default_openai_client(client)           # 全局替换 SDK 的 client
set_default_openai_api("chat_completions")  # DeepSeek 仅兼容 chat_completions
set_tracing_disabled(True)                  # 本地/离线不上传 tracing 日志

def read_jsonl_head(
    file_path: str, 
    lines: int = 3
) -> Dict[str, Any]:
    """
    读取JSONL文件的前几行数据
    
    Args:
        file_path: JSONL文件的路径
        lines: 要读取的行数,默认为3
        
    Returns:
        包含读取结果的字典:
        - success: 是否成功
        - data: 读取的数据列表
        - count: 实际读取的行数
        - error: 错误信息（如果有）
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                "success": False,
                "data": [],
                "count": 0,
                "error": f"文件不存在: {file_path}"
            }
        
        # 读取数据
        data = []
        count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_index, line in enumerate(f):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                try:
                    # 解析JSON行
                    json_data = json.loads(line)
                    data.append(json_data)
                    count += 1
                    
                    # 达到指定行数后停止
                    if count >= lines:
                        break
                        
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "data": data,
                        "count": count,
                        "error": f"第{line_index+1}行JSON解析失败: {str(e)}"
                    }
        
        # 保存结果到上下文
        result = data
        
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "data": [],
            "count": 0,
            "error": f"读取文件失败: {str(e)}"
        }
        return error_result
    

@function_tool
def write_python_code(file_path: str, code: str) -> Dict[str, Any]:
    """
    将Python代码写入文件
    
    Args:
        file_path: 要写入的文件路径
        code: Python代码内容
    
    Returns:
        包含写入结果的字典
    """
    print("写入代码", file_path)
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 写入代码
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        return {
            "success": True,
            "file_path": file_path,
            "message": f"代码已成功写入到 {file_path}"
        }
        
    except Exception as e:
        error_message = str(e)
        
        return {
            "success": False,
            "error": f"写入代码失败: {error_message}"
        }

@function_tool
def execute_python_code(file_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    执行Python代码文件
    
    Args:
        file_path: Python代码文件路径
        args: 命令行参数列表（可选）
    
    Returns:
        包含执行结果的字典
    """
    print("执行代码", file_path)
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"文件不存在: {file_path}"
            }
        
        # 构建命令
        cmd = [sys.executable, file_path]
        if args:
            cmd.extend(args)
        
        # 执行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 获取输出
        stdout, stderr = process.communicate()
        print(stdout, "\n" ,stderr)
        # 根据返回码判断是否成功
        if process.returncode == 0:
            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "message": f"代码成功执行,退出码: {process.returncode}"
            }
        else:
            return {
                "success": False,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "error": f"代码执行失败,退出码: {process.returncode}"
            }
        
    except Exception as e:
        error_message = str(e)
        
        return {
            "success": False,
            "error": f"执行代码时出错: {error_message}"
        }
        
data_download_agent = Agent(
    name="data_download_agent",
    instructions="你是一个专门从huggingface下载数据集的专家助手。你的任务是使用工具编写python代码,根据给定的数据集名称,下载数据集,并保存为指定的格式。",
    tools=[write_python_code, execute_python_code],
    model="gpt-4.1-2025-04-14",
)

async def main(dataset_name, output_path = "dataset_research/workspace/datasets", code_path = "dataset_research/workspace/code"):
    """
    下载数据集
    """
    # print("dataset_name", dataset_name)
    # print("output_path", output_path)
    # print("code_path", code_path)
    # 使用Agent处理
    input=f"""
        首先你需要根据数据集的名称将数据集下载下来,数据集的名称为{dataset_name}
        注意,load_dataset之前一定要先判断是否还需填入子数据文件等相关参数,确保填入的参数符合要求,你的代码必须使用from datasets import get_dataset_config_names的方法实现
        
        将数据集的中的图片信息(key值不一定为'image',且存储图片的方法可能有多种,你需要尽可能考虑多种情况,比如图片可能是二进制文件,可能存在于网上的URL中等等)保存到images文件夹中并对其按照序号命名,确保图片不会被重复命名
        
        修改数据集中的有关图片信息的值为保存后图片所在位置,注意:取dataset的item时,一定要先copy再修改,不要直接修改,否则可能会修改失败。
        
        其它的属性不变,保存至jsonl文件中,注意不要出现这样的问题: "TypeError: Object of type JpegImageFile is not JSON serializable"
        
        你需要对为每个数据集写一个describe.md文件,包含数据集的描述,模态信息,任务名,题型,语言,domain,数量之类的你觉得可能对数据集选择有用的参数,数据集的保存路径为{output_path},具体格式如下所示:
        
        如果数据集的名称为 xxx/xxx 的形式，请你在保存时将文件夹和文件名称转化为 xxx_xxx 的形式
        
        单个config_name时,结构为:
        datasets/                          
        ├── dataset_name/                  # 数据集的名称
        │   ├── describe.md                # 数据集的描述文件
        │   ├── dataset_name.jsonl         # 数据集的JSONL格式数据     
        │   └── images/                    # 数据集的图像子文件夹  
        │       ├── 1.jpg
        │       ├── 2.jpg
        │       └── ...
        └── ... 
        
        多个config_name时,结构为:
        datasets/                          
        ├── dataset_name-config_name1/     # 数据集的名称-子数据集1的名称
        │   ├── describe.md                # 子数据集1的描述文件
        │   ├── config_name1.jsonl         # 子数据集1的JSONL格式数据     
        │   └── images/                    # 子数据集1的图像子文件夹  
        │       ├── 1.jpg
        │       ├── 2.jpg
        │       └── ...
        ├── dataset_name-config_name2/     # 数据集的名称-子数据集2的名称
        │   ├── describe.md                # 子数据集2的描述文件
        │   ├── config_name2.jsonl         # 子数据集2的JSONL格式数据     
        │   └── images/                    # 子数据集2的图像子文件夹  
        │       ├── 1.jpg
        │       ├── 2.jpg
        │       └── ...
        └── ...
        
        所有文件都保存后，你需要将 "dataset_name.jsonl # 数据集的JSONL格式数据" 这个jsonl文件的绝对路径写入 {jsonl_path} 文件的新的一行中，注意不要覆盖原有内容 
        
        代码用python实现, 具体编写代码的流程如下:
        1. 编写完整的Python代码,包含所有必要的导入和错误处理,代码保存路径为保存到{code_path}中。
        2. 你必须使用write_python_code工具将代码保存到data_download.py文件
        3. 你必须使用execute_python_code工具执行该代码
        4. 如果执行遇到bug请修改代码并重新从流程1开始执行,直至代码正常运行
        """
    # print("input", input)
    result = await Runner.run(
        starting_agent=data_download_agent,
        # input="你好",
        input=input,
    )
    return result

if __name__ == "__main__":
    # 设置文件路径
    # metadata = """
    # license: apache-2.0 task_categories: - visual-question-answering language: - en tags: - Vision - medical - biology configs:
    # - config_name: SLAKE data_files: - split: test path: SLAKE/data-.arrow - config_name: VQA_RAD data_files: - split: test path:
    # vqa_rad/data-.arrow - config_name: PathVQA data_files: - split: test path: pathvqa/data-.arrow - config_name: PMC-VQA data_files:
    # - split: test path: pmc_vqa/data-.arrow
    # """
    # dataset_name = "AdaptLLM/biomed-VQA-benchmark"
    dataset_name = "derek-thomas/ScienceQA"
    output_path = "dataset_research/workspace/datasets"
    code_path = "dataset_research/workspace/code"
    # 运行转换
    result = asyncio.run(main(dataset_name, output_path, code_path))
    print(result)