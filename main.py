# 销售话术合规检测系统 - 主程序
# 功能：基于Embedding距离 + 关键词匹配检测销售话术是否违规
# 使用方法：
#   python main.py init --pdf "PDF文件路径"   # 初始化向量库
#   python main.py check --message "待检测的话术"  # 检测话术

import argparse
import json
import os
import sqlite3
import sys
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    BASE_URL, API_KEY, EMBEDDING_MODEL,
    DISTANCE_THRESHOLD, TOP_K, PERSIST_DIR,
    validate_config,
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc", "rules.db")

STOP_WORDS = {"的", "了", "是", "在", "我", "你", "他", "她", "它", "们",
              "这", "那", "有", "和", "与", "或", "不", "也", "都", "就",
              "要", "会", "能", "可以", "可", "把", "被", "让", "给",
              "到", "着", "过", "来", "去", "上", "下", "中", "里", "外"}


def _tokenize(text: str) -> set[str]:
    """中文分词（jieba）+ 英文/数字提取"""
    import jieba
    tokens = set()
    for word in jieba.cut(text):
        word = word.strip()
        if not word or word in STOP_WORDS:
            continue
        # 过滤单字（区分度太低）和纯标点
        if len(word) == 1 and not word.isalnum():
            continue
        tokens.add(word.lower() if word.isascii() else word)
    return tokens


def _keyword_search(message: str) -> list[dict]:
    """从rules.db中查找与输入话术有词汇重叠的规则"""
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

    # 统计词频，过滤在>30%规则中都出现的高频通用词
    token_doc_count = {}
    all_rule_tokens = []
    for row in rows:
        doc_tokens = _tokenize(row[5] or "")
        all_rule_tokens.append((row, doc_tokens))
        for t in doc_tokens:
            token_doc_count[t] = token_doc_count.get(t, 0) + 1
    conn.close()

    high_freq = {t for t, cnt in token_doc_count.items() if cnt > len(rows) * 0.3}
    filtered_query = query_tokens - high_freq
    if not filtered_query:
        return []

    results = []
    for row, doc_tokens in all_rule_tokens:
        rule_id, level, cat1, cat2, standard, violation, penalty = row
        overlap = filtered_query & (doc_tokens - high_freq)
        if not overlap:
            continue

        score = min(1.0, len(overlap) * 0.3)
        category = " ".join(p for p in [level, cat1, cat2] if p) or "未分类"
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
    检测销售话术是否违规（Embedding距离 + 关键词匹配）

    距离(distanee)：Chroma cosine distance，范围[0,2]，0=完全相同，越大越不相关
    阈值(threshold)：距离≤阈值时判定命中，默认0.55（越小越严格）
    """
    # ── 阶段1：Embedding检索 ──────────────────────────────
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, api_key=API_KEY, base_url=BASE_URL
    )

    import time
    for attempt in range(5):
        try:
            db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            db.similarity_search("test", k=1)
            break
        except Exception:
            if attempt < 4:
                time.sleep(2)
            else:
                print("[错误] 向量数据库连接失败，请稍后重试")
                sys.exit(1)

    emb_results = db.similarity_search_with_score(message, k=TOP_K)

    # Chroma返回cosine distance：越小越相似，0=完全相同
    emb_hits = {}
    for doc, distance in emb_results:
        distance = max(0.0, distance)
        rule_id = doc.metadata.get("id", 0)
        emb_hits[rule_id] = {
            "id": rule_id,
            "level": doc.metadata.get("level", ""),
            "category": doc.metadata.get("category", ""),
            "standard": doc.metadata.get("standard", ""),
            "violation": doc.page_content,
            "penalty": doc.metadata.get("penalty", ""),
            "emb_distance": round(distance, 4),
            "keyword_score": 0.0,
        }

    # ── 阶段2：关键词匹配 ─────────────────────────────────
    kw_hits = {r["id"]: r for r in _keyword_search(message)}

    # ── 阶段3：合并评分 ───────────────────────────────────
    # 关键词仅用于给embedding结果加分，不独立判定
    # 综合距离 = embedding距离 - 关键词加分（距离越小越好，加分让距离更小）
    combined = []

    for rule_id, emb in emb_hits.items():
        kw = kw_hits.get(rule_id, {})
        emb_dist = emb.get("emb_distance", 2.0)
        kw_score = kw.get("keyword_score", 0.0)

        # 关键词加分：最多减0.25的距离
        combined_dist = max(0.0, emb_dist - kw_score * 0.25)

        item = {
            "id": rule_id,
            "level": emb.get("level", ""),
            "category": emb.get("category", ""),
            "standard": emb.get("standard", ""),
            "violation": emb.get("violation", ""),
            "penalty": emb.get("penalty", ""),
            "distance": round(combined_dist, 4),
            "emb_distance": round(emb_dist, 4),
            "keyword_score": round(kw_score, 4),
        }
        if kw.get("overlap_words"):
            item["matched_keywords"] = kw["overlap_words"]
        combined.append(item)

    # 关键词独有命中（embedding未召回）的补充
    # 门槛高+虚拟距离高，避免通用词触发误判
    for rule_id, kw in kw_hits.items():
        if rule_id in emb_hits:
            continue
        overlap_count = len(kw.get("overlap_words", []))
        kw_score = kw.get("keyword_score", 0.0)
        if overlap_count >= 3 and kw_score >= 0.9:
            combined_dist = 0.55  # 虚拟距离，仅略低于阈值
        else:
            continue

        item = {
            "id": rule_id,
            "level": kw.get("level", ""),
            "category": kw.get("category", ""),
            "standard": kw.get("standard", ""),
            "violation": kw.get("violation", ""),
            "penalty": kw.get("penalty", ""),
            "distance": round(combined_dist, 4),
            "emb_distance": None,
            "keyword_score": round(kw_score, 4),
        }
        if kw.get("overlap_words"):
            item["matched_keywords"] = kw["overlap_words"]
        combined.append(item)

    # 按距离升序（越小越相似）
    combined.sort(key=lambda x: x["distance"])

    # ── 阶段4：筛选 ──────────────────────────────────────
    violations = [r for r in combined if r["distance"] <= DISTANCE_THRESHOLD]

    is_violation = len(violations) > 0
    response = {
        "is_violation": is_violation,
        "message": message,
        "threshold": DISTANCE_THRESHOLD,
        "hits": violations,
    }

    if is_violation:
        response["alert"] = "检测到违规话术"
    else:
        response["alert"] = "未检测到明显违规"

    return response


def main():
    validate_config()

    parser = argparse.ArgumentParser(description="销售话术合规检测系统")
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
