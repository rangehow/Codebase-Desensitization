
# Codebase Desensitization & Restoration Tool
# 代码脱敏与环境还原工具

本项目包含一套自动化脚本，用于在开源代码前自动识别、提取并替换项目中的敏感硬编码路径（如数据集路径、Checkpoint 路径、用户名等），并利用 LLM 生成语义化的占位符。同时提供还原脚本，帮助新用户快速将代码适配到自己的运行环境。

## 📁 文件说明

*   `desensitize.py`: **脱敏脚本**（仅发布者使用）。扫描代码，利用 LLM 识别路径，替换为 `<PLACEHOLDER>`，并生成配置文件。
*   `restore.py`: **还原脚本**（供使用者使用）。读取配置文件，根据用户填写的路径回填到代码中。
*   `token_config.json`: **配置文件**。存储了占位符的描述、原始值（Debug用）以及用户需要填写的 `target_value`。这个文件会被生成在被脱敏后的仓库根路径下。

---

## 🛠️ Part 1: 如何脱敏 (For Code Authors)

如果你是代码仓库的维护者，准备开源代码，请按照以下步骤操作。

### 1. 环境准备
确保安装了 Python 环境，脚本依赖 `requests` 库调用大模型 API：
```bash
pip install requests
```

### 2. 配置参数
打开 `desensitize.py`，在顶部的 `配置区域` 修改以下变量：

| 变量名 | 说明 | 是否必须 |
| :--- | :--- | :--- |
| `API_KEY` | 智谱 AI (BigModel) 的 API Key。用于调用 GLM 模型生成变量名。 | ✅ 必须 |
| `API_URL` | 大模型 API 地址 (默认为智谱 GLM-4)。 | ✅ 必须 |
| `TARGET_DIR` | **目标仓库路径**。脚本会原地修改该目录下的文件，请务必**提前备份**！ | ✅ 必须 |
| `SENSITIVE_KEYWORDS` | 敏感关键词列表。例如你的用户名、服务器名 (`["xiaoming", "lab_server"]`)。 | 建议配置 |
| `ALLOWED_EXTS` | 需要扫描的文件后缀 (如 `.py`, `.sh`, `.yaml`)。 | 可选 |
| `IGNORE_DIRS` | 忽略扫描的文件夹 (如 `.git`, `node_modules`)。 | 可选 |

### 3. 运行脱敏
```bash
python desensitize.py
```
**运行结果：**
1.  代码中的敏感路径（如 `/home/user/data/coco`）会被替换为占位符（如 `<COCO_DATASET_ROOT>`）。
2.  根目录下会生成 `token_config.json` 文件。

---

## 🚀 Part 2: 如何配置与还原 (For Code Users)

如果你下载了本代码仓库，想要在自己的机器上运行，请按照以下步骤还原路径。

### 1. 填写配置
在项目根目录下找到 `token_config.json`。此时 `target_value` 为空。你需要根据 `description` 的提示，填入你本地环境的真实路径。

**示例 `token_config.json`：**
```json
{
    "tokens": {
        "<HUGGINGFACE_CACHE_DIR>": {
            "description": "Directory where Hugging Face models and datasets are cached.",
            "target_value": "/home/your_name/.cache/huggingface"  <-- 在这里填入你的路径
        },
        "<SSD_LM_LOG_DIR>": {
            "description": "Main directory for storing logs.",
            "target_value": "./logs"  <-- 在这里填入你的路径
        }
    }
}
```

### 2. 运行还原
配置完成后，运行还原脚本：
```bash
python restore.py
```
或者指定目录：
```bash
python restore.py /path/to/your/project
```

脚本会自动将代码中所有的 `<HUGGINGFACE_CACHE_DIR>` 等占位符替换为你填写的路径。

---

## ⚠️ 注意事项

1.  **数据备份**：脱敏脚本 (`desensitize.py`) 会**直接修改源文件**。在运行前请务必提交 Git 或备份代码。
2.  **API 消耗**：脱敏过程会调用 LLM API 进行语义分析，请确保 API Key 有足够的额度。
3.  **还原检查**：`restore.py` 运行后，如果 `token_config.json` 中有未填写的项，脚本会发出警告并询问是否继续。

---

## 📝 详细配置代码段 (Reference)

如果你需要修改脱敏逻辑，请关注 `desensitize.py` 中的以下区域：

```python
# ================= 配置区域 =================

# 1. 大模型配置 (用于生成语义化变量名)
API_KEY = "your_api_key_here" 
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4.5-flash"

# 2. 目标路径 (请修改为你需要脱敏的项目根目录)
TARGET_DIR = "/path/to/your/project"

# 3. 敏感词过滤 (包含这些词的行会被强制扫描)
SENSITIVE_KEYWORDS = ["your_username", "lab_name"]

# 4. 文件扫描规则
ALLOWED_EXTS = {'.py', '.sh', '.yaml', '.json', ...}
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', ...}

# ===========================================
```

---
