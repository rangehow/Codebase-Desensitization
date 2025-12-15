import os
import json
import sys

CONFIG_FILENAME = "token_config.json"

def restore_codebase(root_dir):
    config_path = os.path.join(root_dir, CONFIG_FILENAME)
    
    # 1. 读取配置文件
    if not os.path.exists(config_path):
        print(f"[Fatal] 找不到配置文件: {config_path}")
        return

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
    except Exception as e:
        print(f"[Fatal] 配置文件解析失败: {e}")
        return

    # 2. 解析元数据 (Meta) - 获取操作规则
    meta = full_config.get("meta", {})
    tokens_data = full_config.get("tokens", {})

    # 如果 meta 不存在，说明配置文件版本不对，或者被篡改
    if not meta:
        print("[Error] 配置文件缺少 'meta' 信息，脚本无法确定扫描范围。")
        return

    # 从配置中加载规则
    # set 查找速度更快
    allowed_exts = set(meta.get("allowed_extensions", []))
    ignore_dirs = set(meta.get("ignore_dirs", []))

    print(f"[*] 加载扫描策略: 关注 {len(allowed_exts)} 种文件类型, 忽略 {len(ignore_dirs)} 类目录。")

    # 3. 解析替换数据 (Tokens)
    replacements = {}
    missing_values = []
    
    for placeholder, info in tokens_data.items():
        target_val = info.get("target_value", "").strip()
        if not target_val:
            missing_values.append(placeholder)
        else:
            replacements[placeholder] = target_val

    if missing_values:
        print(f"\n[Warning] 以下占位符未填写 'target_value'，将跳过替换：")
        for mv in missing_values[:5]:
            print(f"  - {mv}")
        if len(missing_values) > 5: print(f"  ... (共 {len(missing_values)} 项)")
        
        # 简单交互，防止用户误操作
        if input("\n确认继续? (y/n): ").lower() != 'y':
            return

    if not replacements:
        print("[Stop] 没有需要替换的内容。请编辑 token_config.json。")
        return

    # 4. 执行替换逻辑
    sorted_placeholders = sorted(replacements.keys(), key=len, reverse=True)
    modified_count = 0
    
    print(f"\n[*] 正在执行回填...")

    for root, dirs, files in os.walk(root_dir):
        # 使用配置中的规则过滤目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        for file in files:
            # 跳过配置文件自身
            if file == CONFIG_FILENAME: continue
            
            # 使用配置中的规则过滤文件后缀
            ext = os.path.splitext(file)[1].lower()
            if ext not in allowed_exts: continue
            
            filepath = os.path.join(root, file)
            
            try:
                # 尝试读取并替换
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                new_content = content
                is_changed = False
                
                for ph in sorted_placeholders:
                    if ph in new_content:
                        new_content = new_content.replace(ph, replacements[ph])
                        is_changed = True
                
                if is_changed:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    modified_count += 1
            except Exception as e:
                # 忽略读取二进制文件或其他错误的报错，保持静默或轻度提示
                pass

    print(f"\n[Success] 任务完成。共修改了 {modified_count} 个文件。")

if __name__ == "__main__":
    target_dir = os.getcwd()
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    restore_codebase(target_dir)