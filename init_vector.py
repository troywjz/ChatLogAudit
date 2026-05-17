# 初始化向量库脚本
# 功能：读取PDF文档，拆分文本，向量化后存储到Chroma数据库
# 使用方法：python init_vector.py --pdf "PDF文件路径"

import argparse
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  # PDF加载器
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本拆分器
from langchain_community.embeddings import OpenAIEmbeddings  # Embedding模型
from langchain_chroma import Chroma  # 向量数据库

# 加载环境变量配置
load_dotenv()

# 从环境变量读取配置
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
PERSIST_DIR = "chroma_db"  # 向量数据库持久化目录


def init_vector_db(pdf_path: str):
    """
    初始化向量数据库

    参数:
        pdf_path: PDF文件路径

    处理流程:
        1. 加载PDF文档
        2. 拆分文本为小块（保留条款完整性）
        3. 调用Embedding模型生成向量
        4. 存储到Chroma数据库
    """
    # 检查PDF文件是否存在
    if not os.path.exists(pdf_path):
        print(f"[错误] PDF文件不存在: {pdf_path}")
        return

    print(f"[INFO] 正在加载PDF: {pdf_path}")

    # 步骤1：加载PDF文档
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[INFO] PDF加载完成，共 {len(docs)} 页")

    # 步骤2：拆分文本
    # 设计原则：
    #   - 优先按段落拆分(\n\n)，保持条款完整性
    #   - chunk_size=500 保证单个条款完整，避免跨条款语义混杂
    #   - chunk_overlap=50 允许轻微重叠，防止边界信息丢失
    print(f"[INFO] 正在拆分文档文本...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", " "],  # 按优先级：段落 > 换行 > 句号 > 空格
        chunk_size=500,      # 每个文本块的最大字符数
        chunk_overlap=50     # 相邻块之间的重叠字符数
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] 文本拆分完成，共 {len(chunks)} 个文本块")

    # 步骤3：初始化Embedding模型
    # 使用OpenAI兼容的API（硅基流动等）
    print(f"[INFO] 正在初始化Embedding模型: {EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 步骤4：生成向量并存储
    print(f"[INFO] 正在生成向量并存储到Chroma数据库...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"[SUCCESS] 向量库初始化完成！存储目录: {PERSIST_DIR}")
    print(f"[INFO] 共存储 {db._collection.count()} 条向量记录")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="销售话术合规检测系统 - 向量库初始化工具"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="PDF文件路径（支持相对路径或绝对路径）"
    )
    args = parser.parse_args()

    # 执行初始化
    init_vector_db(args.pdf)