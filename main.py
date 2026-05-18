# 销售话术合规检测系统 - 主程序
# 功能：基于Embedding语义相似度 + 关键词匹配检测销售话术是否违规
# 使用方法：
#   python main.py init --pdf "PDF文件路径"   # 初始化向量库
#   python main.py check --message "待检测的话术"  # 检测话术

import argparse
import json
import os
import re
import sqlite3
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    BASE_URL, API_KEY, EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD, TOP_K, PERSIST_DIR,
    validate_config,
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc", "rules.db")

# 中文停用词（检测时忽略的高频词）
STOP_WORDS = {"的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
              "这", "那", "有", "和", "与", "或", "不", "也", "都", "就",
              "要", "会", "能", "可以", "可", "会", "把", "被", "让", "给",
              "到", "着", "过", "来", "去", "上", "下", "中", "里", "外"}


def _tokenize(text: str) -> set[str]:
    """简易中文分词：提取连续中文字符和英文/数字片段"""
    tokens = set()
    # 提取连续中文片段（2字以上）
    for m in re.finditer(r'[\u4e00-\u9fff]{2,}', text):
        word = m.group()
        # 2-gram 切分
        for i in range(len(word) - 1):
            tokens.add(word[i:i + 2])
        # 整个词也加入
        if len(word) >= 2:
            tokens.add(word)
    # 提取英文+数字片段
    for m in re.finditer(r'[a-zA-Z0-9%]+', text):
        tokens.add(m.group().lower())
    # 过滤停用词
    tokens = {t for t in tokens if t not in STOP_WORDS}
    return tokens


def _keyword_search(message: str) -> list[dict]:
    """基于关键词重叠的检索：从rules.db中查找与输入话术有词汇重叠的规则"""
    if not os.path.exists(DB_PATH):
        return []

    query_tokens = _tokenize(message)
    if not query_tokens:
        return []

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        'SELECT rule_id, level, category1, category2, standard, violation, penalty FROM rules ORDER BY rule_id'
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        rule_id, level, cat1, cat2, standard, violation, penalty = row
        # 合并所有文本用于关键词匹配
        full_text = f"{standard or ''} {violation or ''}"
        doc_tokens = _tokenize(full_text)

        # 计算关键词重叠率
        overlap = query_tokens & doc_tokens
        if not overlap:
            continue

        # 按匹配词数量计分（1词=0.3, 2词=0.6, 3+词=1.0），避免噪声词稀释
        score = min(1.0, len(overlap) * 0.3)
        if score < 0.15:
            continue

        category_parts = [p for p in [level, cat1, cat2] if p]
        category = " ".join(category_parts) if category_parts else "未分类"

        results.append({
            "id": rule_id,
            "level": level or "",
            "category": category,
            "standard": standard or "",
            "violation": violation or "",
            "penalty": penalty or "",
            "keyword_score": round(score, 4),
            "overlap_words": sorted(overlap),
        })

    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results[:TOP_K]


def check_message(message: str) -> dict:
    """
    检测销售话术是否违规（混合检索：Embedding语义 + 关键词匹配）

    参数:
        message: 待检测的销售话术文本

    返回:
        dict: 包含检测结果的字典

    算法流程:
        1. Embedding语义检索：在Chroma中检索Top-K最相似的规则
        2. 关键词匹配检索：从rules.db中查找词汇重叠的规则
        3. 合并去重：取两种检索的并集，按综合得分排序
        4. 筛选出得分超过阈值的结果
        5. 返回结构化JSON结果
    """
    # ── 阶段1：Embedding语义检索 ─────────────────────────
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    import time
    for attempt in range(5):
        try:
            db = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            db.similarity_search("test", k=1)
            break
        except Exception:
            if attempt < 4:
                time.sleep(2)
            else:
                print("[错误] 向量数据库连接失败，请稍后重试")
                print("[提示] 如果刚运行过 init 命令，请等待几秒后再试")
                sys.exit(1)

    emb_results = db.similarity_search_with_score(message, k=TOP_K)

    # 转换为统一格式
    emb_hits = {}
    for doc, score in emb_results:
        similarity = max(0.0, min(1.0, 1 - score))
        rule_id = doc.metadata.get("id", 0)
        emb_hits[rule_id] = {
            "id": rule_id,
            "level": doc.metadata.get("level", ""),
            "category": doc.metadata.get("category", ""),
            "standard": doc.metadata.get("standard", ""),
            "violation": doc.page_content,
            "penalty": doc.metadata.get("penalty", ""),
            "embedding_score": round(similarity, 4),
            "keyword_score": 0.0,
        }

    # ── 阶段2：关键词匹配检索 ───────────────────────────
    kw_results = _keyword_search(message)
    kw_hits = {}
    for r in kw_results:
        kw_hits[r["id"]] = r

    # ── 阶段3：合并去重 + 综合评分 ──────────────────────
    all_ids = set(emb_hits.keys()) | set(kw_hits.keys())
    combined = []

    for rule_id in all_ids:
        emb = emb_hits.get(rule_id, {})
        kw = kw_hits.get(rule_id, {})

        emb_score = emb.get("embedding_score", 0.0)
        kw_score = kw.get("keyword_score", 0.0)
        overlap_count = len(kw.get("overlap_words", []))

        # 综合评分策略：
        # - embedding+keyword 双命中：取max × 1.1（双重确认加分）
        # - 仅keyword命中：需至少2个关键词重叠才采信（防误判）
        # - 仅embedding命中：直接使用embedding分数
        if emb_score > 0 and kw_score > 0:
            combined_score = max(emb_score, kw_score) * 1.1
        elif kw_score > 0 and overlap_count >= 2:
            combined_score = kw_score
        elif emb_score > 0:
            combined_score = emb_score
        else:
            continue  # 仅1个关键词匹配且无embedding支持，跳过

        combined_score = min(1.0, combined_score)

        # 优先用embedding的结构化数据（更完整），否则用keyword的
        base = emb if emb else kw
        item = {
            "id": rule_id,
            "level": base.get("level", ""),
            "category": base.get("category", ""),
            "standard": base.get("standard", ""),
            "violation": base.get("violation", ""),
            "penalty": base.get("penalty", ""),
            "similarity": round(combined_score, 4),
            "embedding_score": round(emb_score, 4),
            "keyword_score": round(kw_score, 4),
        }

        if kw.get("overlap_words"):
            item["matched_keywords"] = kw["overlap_words"]

        combined.append(item)

    # 按综合得分降序排列
    combined.sort(key=lambda x: x["similarity"], reverse=True)

    # ── 阶段4：筛选命中结果 ────────────────────────────
    violations = [r for r in combined if r["similarity"] >= SIMILARITY_THRESHOLD]

    is_violation = len(violations) > 0
    violation_rules = [v["violation"] for v in violations]

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
    validate_config()

    parser = argparse.ArgumentParser(
        description="销售话术合规检测系统"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="初始化向量库")
    init_input = init_parser.add_mutually_exclusive_group(required=True)
    init_input.add_argument("--pdf", help="PDF文件路径")
    init_input.add_argument("--txt", help="TXT文本文件路径")

    check_parser = subparsers.add_parser("check", help="检测销售话术是否违规")
    check_parser.add_argument("--message", required=True, help="待检测的销售话术")

    args = parser.parse_args()

    if args.command == "init":
        from init_vector import init_vector_db
        init_vector_db(args.pdf or args.txt)

    elif args.command == "check":
        result = check_message(args.message)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
