import os
import re
import json
import time
import concurrent.futures
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Any, Tuple

# === 修改点 1: 引入 OpenAI SDK (DeepSeek 兼容) ===
from openai import OpenAI

# ================= 配置区域 =================

# === 修改点 2: 配置 DeepSeek ===
# 这里填入你的 Key


# DeepSeek 的 Base URL
BASE_URL = "https://api.deepseek.com"

# 模型名称
MODEL_NAME = "deepseek-chat"

# 目标仓库路径 (保持不变)
TARGET_DIR = r"C:\Users\lenovo\Downloads\mtr\mtr"

# 输出的配置文件名
CONFIG_OUTPUT_FILE = "token_config.json"

# 每个 Key 允许的并发线程数。
# DeepSeek 并发通常还可以，只有1个Key的情况下，为了提高速度建议设为 5-10，
# 但要注意不要触发 Rate Limit (429错误)
THREADS_PER_KEY = 5 
# 总并发数自动计算
TOTAL_MAX_WORKERS = len(API_KEYS) * THREADS_PER_KEY

ALLOWED_EXTS = {'.py', '.sh', '.yaml', '.yml', '.json', '.md', '.txt', '.conf', '.ini', '.xml', '.properties'}
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', '.idea', '.vscode', 'dist', 'build', 'target'}

SENSITIVE_KEYWORDS = ["rangehow"]

REGEX_UNIX_STR = r'(?:["\']|\s|^)(/[\w\-\.]+(?:/[\w\-\.]+)+)(?:["\']|\s|$)'
REGEX_WIN_STR = r'(?:["\']|\s|^)([a-zA-Z]:\\[\w\-\.]+(?:\\[\w\-\.]+)+)(?:["\']|\s|$)'

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
    chunk_size = (total_files // num_processes) + 1 if total_files > 0 else 1
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

# ================= LLM 处理核心 (修改为 OpenAI/DeepSeek 版) =================

def call_llm_for_placeholder(original_item: str, item_contexts_str: str, assigned_api_key: str) -> Tuple[str, str]:
    """
    使用 OpenAI SDK 调用 DeepSeek API。
    """
    # === 修改点 3: 初始化 OpenAI 客户端 ===
    client = OpenAI(
        api_key=assigned_api_key,
        base_url=BASE_URL
    )
    
    # DeepSeek 的 JSON 模式要求 Prompt 中必须包含 "json" 字样
    prompt = f"""
    Code Desensitization Task.

    Target String: "{original_item}"
    
    Context (Target is wrapped in [[ ... ]]):
    ```
    {item_contexts_str}
    ```
    
    Instructions:
    1. Identify the semantic meaning of the Target String (e.g. path, secret, ip).
    2. Create a UNIQUE, UPPERCASE_SNAKE_CASE placeholder wrapped in < > (e.g., <SAR8_DATASET_ROOT>).
    3. Write a short description.
    
    Output Format: JSON ONLY.
    {{
        "placeholder": "<YOUR_PLACEHOLDER>",
        "description": "Short explanation."
    }}
    """
    
    max_retries = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            # === 修改点 4: 使用 OpenAI 风格发起请求 ===
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a code assistant. You ONLY respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={'type': 'json_object'},  # 强制 JSON 模式
                temperature=0.1,
                timeout=30  # 设置超时
            )
            
            # === 修改点 5: 响应解析 ===
            content = response.choices[0].message.content
            
            # 虽然开了 JSON mode，但为了保险起见，还是做一下清洗（去掉可能的 Markdown 标记）
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "")
            content = content.strip()
            
            try:
                data = json.loads(content)
                placeholder = data.get("placeholder")
                description = data.get("description", "No description provided.")
                
                if not placeholder:
                    raise ValueError("JSON parsed but 'placeholder' is empty.")
                
                # 格式规范化
                if not placeholder.startswith("<"): placeholder = f"<{placeholder}"
                if not placeholder.endswith(">"): placeholder = f"{placeholder}>"
                
                return placeholder, description

            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response: {content[:50]}...")

        except Exception as e:
            last_exception = e
            key_hint = assigned_api_key[-4:]
            print(f"[Warn] Item '{original_item[:15]}...' failed on Key(...{key_hint}) Attempt {attempt+1}: {e}")
            
            # 简单的退避策略
            error_str = str(e)
            if "429" in error_str: # Rate Limit
                time.sleep(3 * (attempt + 1))
            else:
                time.sleep(2)
    
    # 如果重试完还是失败，抛出错误
    raise RuntimeError(f"ALL RETRIES FAILED for item '{original_item}'. Last error: {last_exception}")

def generate_mappings_and_config(sensitive_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """
    并发请求 LLM，如果有任何一个任务失败，主进程立刻报错退出。
    """
    mapping = {} 
    config_data = {} 
    used_placeholders = set() 

    # 准备任务列表
    llm_tasks = []
    for item, data in sensitive_data.items():
        unique_contexts = sorted(list(data["contexts"]))[:2]
        joined_context = "\n...SNIP...\n".join(unique_contexts)
        llm_tasks.append((item, joined_context))
    
    num_tasks = len(llm_tasks)
    num_keys = len(API_KEYS)
    
    print(f"[*] 准备处理 {num_tasks} 个敏感项。")
    print(f"[*] 可用 Key 数量: {num_keys}，总并发线程数: {TOTAL_MAX_WORKERS}")
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=TOTAL_MAX_WORKERS) as executor:
            future_to_item = {}
            
            for i, (item, context) in enumerate(llm_tasks):
                selected_key = API_KEYS[i % num_keys]
                future = executor.submit(call_llm_for_placeholder, item, context, selected_key)
                future_to_item[future] = item
            
            print(f"[*] 已提交所有任务，正在执行...")
            
            processed = 0
            for future in concurrent.futures.as_completed(future_to_item):
                original_item = future_to_item[future]
                
                try:
                    placeholder, description = future.result()
                except Exception as e:
                    print(f"\n[!!!] 致命错误：处理 '{original_item}' 失败。")
                    print(f"[!!!] 原因: {e}")
                    print("[!!!] 正在终止所有任务...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise e

                # 后续处理成功逻辑
                base_placeholder = placeholder
                counter = 2
                while placeholder in used_placeholders:
                    placeholder = base_placeholder.replace(">", f"_V{counter}>")
                    counter += 1
                
                used_placeholders.add(placeholder)
                mapping[original_item] = placeholder
                
                config_data[placeholder] = {
                    "description": description,
                    "target_value": "",  
                }
                
                processed += 1
                if processed % 5 == 0:
                    print(f"    [{processed}/{num_tasks}] processed ok.")

    except Exception as fatal_error:
        print("\n[FAILED] 程序因错误而中断。未生成配置文件，未修改文件。")
        raise fatal_error

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
    final_output = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "allowed_extensions": list(ALLOWED_EXTS),
            "ignore_dirs": list(IGNORE_DIRS)
        },
        "tokens": {}
    }

    sorted_keys = sorted(config_data.keys())
    for k in sorted_keys:
        final_output["tokens"][k] = config_data[k]
        
    output_path = os.path.join(TARGET_DIR, CONFIG_OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"\n[SUCCESS] 汇总配置文件已生成: {output_path}")

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        print(f"错误：目录 {TARGET_DIR} 不存在")
    else:
        # 1. 扫描
        sensitive_data = find_sensitive_items_parallel(TARGET_DIR)
        
        if sensitive_data:
            # 2. 生成映射和配置 (出错会直接 Crash)
            mapping, config_data = generate_mappings_and_config(sensitive_data)
            
            # 3. 替换代码 (只有上面成功了才会执行到这里)
            apply_replacements(TARGET_DIR, mapping)
            
            # 4. 生成汇总文件
            save_config_file(config_data)
        else:
            print("[*] 未发现敏感项。")