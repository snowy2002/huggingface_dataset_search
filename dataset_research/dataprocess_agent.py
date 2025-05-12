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
        lines: 要读取的行数，默认为3
        
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
        
        # 根据返回码判断是否成功
        if process.returncode == 0:
            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "message": f"代码成功执行，退出码: {process.returncode}"
            }
        else:
            return {
                "success": False,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": process.returncode,
                "error": f"代码执行失败，退出码: {process.returncode}"
            }
        
    except Exception as e:
        error_message = str(e)
        
        return {
            "success": False,
            "error": f"执行代码时出错: {error_message}"
        }
        
jsonl_reader_agent = Agent(
    name="JSONL预览工具",
    instructions="你是一个专门将一种JSONL格式转换为另一种格式的专家助手。你的任务是将target.jsonl文件中的数据转换为与example.jsonl文件格式匹配的数据。",
    tools=[write_python_code, execute_python_code],
    model="gpt-4.1-2025-04-14",
)

async def main(raw_file_path, example_file_path = "dataset_research/workspace/datasets/dataset1_example/dataset1.jsonl", code_path = "dataset_research/workspace/code"):
    target_data = read_jsonl_head(raw_file_path)
    example_data = read_jsonl_head(example_file_path)
    
    print("target_data", target_data)
    print("example_data", example_data)
    
    # 使用Agent处理
    result = await Runner.run(
        starting_agent=jsonl_reader_agent,
        input=f"""
        target.jsonl文件中的数据前三行为{target_data}, example.jsonl文件中的数据前三行为{example_data}。
        
        请先分析两个文件中的数据格式，分析example中每一个key的内容，并从target中每一个key的内容中寻找最合适的匹配方法。
        然后根据这种匹配关系，编写格式转化代码，将target文件中的所有数据转化成符合example中格式要求的文件。
        
        在分析完成后，请执行以下步骤：
        1. 编写完整的Python代码，包含所有必要的导入和错误处理，能够实现从{raw_file_path}中读取数据，并转换为{example_file_path}中的格式，重新保存回原文件{raw_file_path}中。
        2. 使用write_python_code工具将代码保存到format.py文件,保存路径为{code_path}
        3. 使用execute_python_code工具执行该代码，将转换后的数据保存到原文件{raw_file_path}中
        
        确保代码能够正确处理各种边缘情况，并提供详细的执行结果报告。
        """,
    )
    return result

if __name__ == "__main__":
    # 设置文件路径
    raw_file_path = "dataset_research/workspace/datasets/leonardPKU_ScienceQA_Test_IMG-default/default.jsonl"
    example_file_path = "dataset_research/workspace/datasets/dataset1_example/dataset1.jsonl"
    code_path = "dataset_research/workspace/code"
    
    # 运行转换
    result = asyncio.run(main(raw_file_path, example_file_path, code_path))
    print(result)