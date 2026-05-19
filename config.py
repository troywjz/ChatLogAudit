import sys
import os
from dotenv import load_dotenv

load_dotenv()

# UTF-8 输出（修复 Windows 终端中文乱码）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# API 配置
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# VLM 视觉模型配置（用于 PDF 表格识图）
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# 检测参数
# DISTANCE_THRESHOLD: cosine distance阈值，≤此值判定命中。0=完全相同，越大越不相关
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.55"))
TOP_K = int(os.getenv("TOP_K", "3"))
PERSIST_DIR = "chroma_db"


def validate_config():
    """校验必要的环境变量配置，缺失时给出明确提示"""
    missing = []
    if not BASE_URL:
        missing.append("BASE_URL")
    if not API_KEY:
        missing.append("API_KEY")
    if not EMBEDDING_MODEL:
        missing.append("EMBEDDING_MODEL")
    if missing:
        print(f"[错误] 缺少必要的环境变量: {', '.join(missing)}")
        print("[提示] 请检查 .env 文件是否正确配置，可参考 .env.example")
        sys.exit(1)
