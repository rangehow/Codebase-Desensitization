import os
import re
import json
import time
import requests
import concurrent.futures
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Any, Tuple
import requests.exceptions

# ================= 配置区域 =================

API_KEY = "b8f0d3baad112a07ffff399c6e14f6e6.9vtIdrPqiWiJIkJX" 
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4.5-flash"

# 目标仓库路径
TARGET_DIR = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/fuck/bd3lms"

# 输出的配置文件名
CONFIG_OUTPUT_FILE = "token_config.json"

# 这些配置现在会被写入 JSON，指导回填脚本如何工作
ALLOWED_EXTS = {'.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt', '.conf', '.ini', '.xml', '.properties'}
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', '.idea', '.vscode', 'dist', 'build', 'target'}

SENSITIVE_KEYWORDS = ["rangehow"]
safe_max_workers = 1 

REGEX_UNIX_STR = r'(?:"|\')(/[\w\-\.]+(?:/[\w\-\.]+)+)(?:"|\')'
REGEX_WIN_STR = r'(?:"|\')([a-zA-Z]:\\[\w\-\.]+(?:\\[\w\-\.]+)+)(?:"|\')'

# ===========================================

def is_allowed_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return ext in ALLOWED_EXTS

def scan_file_content(filepath) -> Dict[str, Dict[str, Any]]:
    local_results = {}
    if not is_allowed_file(filepath):
        return local_results

    try:
        rel_path = os.path.relpath(filepath, TARGET_DIR)
    except ValueError:
        rel_path = filepath 

    try:
        pattern_unix = re.compile(REGEX_UNIX_STR)
        pattern_win = re.compile(REGEX_WIN_STR)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if len(line) > 2000: continue

            found_items = set()
            matches_path = []
            matches_path.extend(pattern_unix.findall(line))
            matches_path.extend(pattern_win.findall(line))
            for p in matches_path: found_items.add(p)

            for keyword in SENSITIVE_KEYWORDS:
                if keyword in line: found_items.add(keyword)

            if found_items:
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context_snippet = "".join(lines[start:end]).strip()
                
                for item in found_items:
                    # 标记上下文
                    marked_context = context_snippet.replace(item, f"[[ {item} ]]")
                    
                    if item not in local_results:
                        local_results[item] = {
                            "contexts": {marked_context}, 
                            "files": {rel_path}
                        }
                    else:
                        local_results[item]["files"].add(rel_path)
                        local_results[item]["contexts"].add(marked_context)
    except Exception:
        pass
    return local_results

def worker_process_batch(file_list: List[str]) -> Dict[str, Dict[str, Any]]:
    batch_results = {}
    for filepath in file_list:
        file_res = scan_file_content(filepath)
        for item, data in file_res.items():
            if item not in batch_results:
                batch_results[item] = data
            else:
                batch_results[item]["files"].update(data["files"])
                batch_results[item]["contexts"].update(data["contexts"])
    return batch_results

def find_sensitive_items_parallel(root_dir) -> Dict[str, Dict[str, Any]]:
    target_files = []
    print(f"[*] 正在扫描目录: {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            filepath = os.path.join(root, file)
            if is_allowed_file(filepath):
                target_files.append(filepath)

    total_files = len(target_files)
    print(f"[*] 符合白名单的文件共 {total_files} 个。")
    if total_files == 0: return {}

    num_processes = min(cpu_count(), 8)
    chunk_size = (total_files // num_processes) + 1
    chunks = [target_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    final_results = {}
    print(f"[*] 启动 {num_processes} 个进程进行并行扫描...")
    with Pool(processes=num_processes) as pool:
        results_list = pool.map(worker_process_batch, chunks)
    
    for res in results_list:
        for item, data in res.items():
            if item not in final_results:
                final_results[item] = data
            else:
                final_results[item]["files"].update(data["files"])
                final_results[item]["contexts"].update(data["contexts"])
    
    print(f"[*] 扫描结束。共发现 {len(final_results)} 个唯一的敏感项。")
    return final_results

def call_llm_for_placeholder(original_item: str, item_contexts_str: str) -> Tuple[str, str]:
    """
    请求 LLM 生成占位符和描述。
    返回: (placeholder, description)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 强制要求 JSON 格式输出
    prompt = f"""
    Code Desensitization Task.

    Target String: "{original_item}"
    
    Context (Target is wrapped in [[ ... ]]):
    ```
    {item_contexts_str}
    ```
    
    Instructions:
    1. Identify the semantic meaning of the Target String (e.g., dataset path, checkpoint dir, username, output dir).
    2. Create a UNIQUE, UPPERCASE_SNAKE_CASE placeholder wrapped in < > (e.g., <SAR8_DATASET_ROOT>).
    3. Write a short, natural language description explaining what this path represents so a new user knows what to fill in.
    
    Output Format: JSON ONLY.
    {{
        "placeholder": "<YOUR_PLACEHOLDER>",
        "description": "Short explanation of what this path is."
    }}
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a code assistant. You ONLY respond with valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                res_json = response.json()
                content = res_json['choices'][0]['message']['content'].strip()
                
                # 清洗 Markdown 代码块标记
                content = content.replace("```json", "").replace("```", "").strip()
                
                try:
                    data = json.loads(content)
                    placeholder = data.get("placeholder", "<UNKNOWN_PLACEHOLDER>")
                    description = data.get("description", "No description provided.")
                    
                    # 再次清洗占位符格式
                    if not placeholder.startswith("<"): placeholder = f"<{placeholder}"
                    if not placeholder.endswith(">"): placeholder = f"{placeholder}>"
                    
                    return placeholder, description
                except json.JSONDecodeError:
                    # 如果解析失败，简单兜底
                    clean_content = re.sub(r'[^A-Z0-9_]', '', content.upper())
                    return f"<{clean_content}>", "Complex string, check code context."

            elif response.status_code == 429:
                sleep_time = 2 * (2 ** attempt)
                time.sleep(sleep_time)
                continue
            else:
                raise Exception(f"API Error: {response.status_code}")

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return f"<UNKNOWN_{int(time.time())}>", "Generation failed."
    
    return f"<UNKNOWN_{int(time.time())}>", "Generation failed."

def generate_mappings_and_config(sensitive_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    生成映射关系，并构建详细的配置信息
    """
    mapping = {} # 原文 -> 占位符 (用于代码替换)
    config_data = {} # 占位符 -> {描述, 用户填写的字段}
    
    used_placeholders = set() 

    llm_tasks = []
    for item, data in sensitive_data.items():
        unique_contexts = sorted(list(data["contexts"]))[:2] # 上下文只取2条
        joined_context = "\n...SNIP...\n".join(unique_contexts)
        llm_tasks.append((item, joined_context))
    
    print(f"[*] 请求大模型生成语义信息 (并发数: {safe_max_workers})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=safe_max_workers) as executor:
        future_to_item = {
            executor.submit(call_llm_for_placeholder, item, context): item 
            for item, context in llm_tasks
        }
        
        processed = 0
        total = len(sensitive_data)
        
        for future in concurrent.futures.as_completed(future_to_item):
            original_item = future_to_item[future]
            try:
                placeholder, description = future.result()
                
                # 防撞逻辑
                base_placeholder = placeholder
                counter = 2
                while placeholder in used_placeholders:
                    placeholder = base_placeholder.replace(">", f"_V{counter}>")
                    counter += 1
                
                used_placeholders.add(placeholder)
                
                # 存入替换映射
                mapping[original_item] = placeholder
                
                # 存入配置文件结构
                # 注意：这里我们以占位符为Key，方便用户填写
                config_data[placeholder] = {
                    "description": description,
                    "target_value": "",  # 用户之后要填这个
                }
                
                processed += 1
                if processed % 5 == 0:
                    print(f"    [{processed}/{total}] item processed.")
            
            except Exception as e:
                print(f"Error processing {original_item}: {e}")

    return mapping, config_data

def apply_replacements(root_dir, mapping: Dict[str, str]):
    if not mapping: return
    # 按长度倒序，避免部分匹配
    sorted_items = sorted(mapping.keys(), key=len, reverse=True)
    
    print(f"[*] 正在替换文件内容...")
    count = 0
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            filepath = os.path.join(root, file)
            if not is_allowed_file(filepath): continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                new_content = content
                is_modified = False
                
                for original_item in sorted_items:
                    if original_item in new_content:
                        new_content = new_content.replace(original_item, mapping[original_item])
                        is_modified = True
                
                if is_modified:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    count += 1
            except: pass
    print(f"[*] 替换完成，共修改了 {count} 个文件。")

def save_config_file(config_data: Dict[str, Any]):
    """
    保存用户配置文件，包含元数据(Meta)和Token数据
    """
    # 构造最终的 JSON 结构
    final_output = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "allowed_extensions": list(ALLOWED_EXTS), # 转为 list 以便 JSON 序列化
            "ignore_dirs": list(IGNORE_DIRS)
        },
        "tokens": {}
    }

    # 对 Token 按字母排序
    sorted_keys = sorted(config_data.keys())
    for k in sorted_keys:
        final_output["tokens"][k] = config_data[k]
        
    output_path = os.path.join(TARGET_DIR, CONFIG_OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n[SUCCESS] 汇总配置文件已生成: {output_path}")
    print(f"[*] 已将扫描策略（{len(ALLOWED_EXTS)}种后缀, {len(IGNORE_DIRS)}个忽略目录）写入配置文件。")
if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        print(f"错误：目录 {TARGET_DIR} 不存在")
    else:
        # 1. 扫描
        sensitive_data = find_sensitive_items_parallel(TARGET_DIR)
        
        if sensitive_data:
            # 2. 生成映射和配置
            mapping, config_data = generate_mappings_and_config(sensitive_data)
            
            # 3. 替换代码
            apply_replacements(TARGET_DIR, mapping)
            
            # 4. 生成汇总文件
            save_config_file(config_data)
        else:
            print("[*] 未发现敏感项。")