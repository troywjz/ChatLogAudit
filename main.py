# 销售话术合规检测系统 - 主程序
# 功能：基于Embedding语义相似度检测销售话术是否违规
# 使用方法：
#   python main.py init --pdf "PDF文件路径"   # 初始化向量库
#   python main.py check --message "待检测的话术"  # 检测话术

import argparse
import json
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings  # Embedding模型
from langchain_chroma import Chroma  # 向量数据库

# 加载环境变量配置
load_dotenv()

# 从环境变量读取检测参数
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.72"))  # 默认阈值0.72
TOP_K = int(os.getenv("TOP_K", "3"))  # 默认返回Top-3最相似结果
PERSIST_DIR = "chroma_db"  # 向量数据库目录


def check_message(message: str) -> dict:
    """
    检测销售话术是否违规

    参数:
        message: 待检测的销售话术文本

    返回:
        dict: 包含检测结果的字典
            - is_violation: bool, 是否检测到违规
            - message: str, 输入的话术
            - threshold: float, 使用的相似度阈值
            - hits: list, 命中的规则列表（每条包含content/similarity/source/page）
            - alert: str, 警告信息
            - penalty_hint: str, 处罚提示（仅违规时返回）

    算法流程:
        1. 将输入话术通过Embedding模型转为向量
        2. 在Chroma中检索Top-K最相似的规则文本
        3. 将距离转换为相似度（1 - distance）
        4. 筛选出相似度超过阈值的结果
        5. 返回结构化JSON结果
    """
    # 初始化Embedding模型
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # 加载向量数据库
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    # 执行相似度检索
    # 注意：Chroma返回的是cosine距离（越小越相似），需转换为相似度
    results = db.similarity_search_with_score(message, k=TOP_K)

    # 处理检索结果
    violations = []
    for doc, score in results:
        # Chroma 0.4+ 返回距离（distance），需要转换为相似度
        # 距离范围通常是 [0, ∞)，0表示完全相同
        similarity = 1 - score

        # 检查是否超过阈值
        if similarity >= SIMILARITY_THRESHOLD:
            violations.append({
                "content": doc.page_content,  # 命中的规则原文
                "similarity": round(similarity, 4),  # 保留4位小数
                "source": doc.metadata.get("source", "未知"),
                "page": doc.metadata.get("page", 0)
            })

    # 构建返回结果
    is_violation = len(violations) > 0
    violation_rules = [v["content"] for v in violations]

    response = {
        "is_violation": is_violation,
        "message": message,
        "threshold": SIMILARITY_THRESHOLD,
        "hits": violations,
        "violation_rules": violation_rules
    }

    if is_violation:
        response["alert"] = "检测到违规话术"
        response["penalty_hint"] = (
            "请参照上述命中的规则原文，"
            "结合《销售话术管理规定》处罚条款执行"
        )
    else:
        response["alert"] = "未检测到明显违规"
        response["penalty_hint"] = None

    return response


def main():
    """主函数：解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(
        description="销售话术合规检测系统"
    )

    # 添加子命令：init（初始化）和 check（检测）
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init 子命令：初始化向量库
    init_parser = subparsers.add_parser("init", help="初始化向量库")
    init_parser.add_argument(
        "--pdf",
        required=True,
        help="PDF文件路径"
    )

    # check 子命令：检测话术
    check_parser = subparsers.add_parser("check", help="检测销售话术是否违规")
    check_parser.add_argument(
        "--message",
        required=True,
        help="待检测的销售话术（建议用引号包裹）"
    )

    args = parser.parse_args()

    # 根据子命令执行相应操作
    if args.command == "init":
        # 导入并调用初始化函数
        from init_vector import init_vector_db
        init_vector_db(args.pdf)

    elif args.command == "check":
        # 执行话术检测
        result = check_message(args.message)
        # 输出格式化JSON结果
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()