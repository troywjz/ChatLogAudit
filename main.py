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
    BASE_URL, API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    DISTANCE_THRESHOLD, TOP_K, PERSIST_DIR,
    validate_config,
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc", "rules.db")

# 违规触发词字典：仅收录在销售话术中几乎必定表示违规的词/短语
# 键=触发词，值=最相关的规则ID（可多条）
VIOLATION_TRIGGERS: dict[str, list[int]] = {
    # C类-过度承诺
    "包过": [36], "100%": [36], "保底": [36, 37], "保证通过": [36],
    "一次过": [36], "直接发证": [36], "肯定能过": [36],
    "挂靠": [40], "挂靠费": [40],
    "包就业": [37], "安排工作": [37], "保底薪资": [37],
    "办假证": [45], "不用考试": [46], "不用学习": [46],
    "随时退款": [48], "7天无理由": [48], "不想学直接退": [48],
    "最大": [44], "第一": [44], "唯一": [44], "首家": [44],
    "虚假宣传": [44],
    # A类-损害公司利益
    "私收": [6], "私自收费": [6], "直接转我": [6],
    "不用还": [13], "不用还款": [13], "免还款": [13],
    "代退费": [20], "帮你退": [20], "帮你操作": [20],
    "不走公司": [6, 20], "绕过公司": [6, 20],
}

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
        if len(word) == 1 and not word.isalnum():
            continue
        tokens.add(word.lower() if word.isascii() else word)
    return tokens


def _find_trigger_matches(message: str) -> list[dict]:
    """查找违规触发词匹配

    使用人工审核的触发词字典，匹配销售话术中几乎必定表示违规的词/短语。
    这些词在正常合规话术中不会出现，因此匹配即可确信违规。
    """
    if not os.path.exists(DB_PATH):
        return []

    # 收集所有命中的触发词及对应规则ID
    triggered_rules = {}  # rule_id → [matched_words]
    for trigger, rule_ids in VIOLATION_TRIGGERS.items():
        if trigger in message:
            for rid in rule_ids:
                if rid not in triggered_rules:
                    triggered_rules[rid] = []
                triggered_rules[rid].append(trigger)

    if not triggered_rules:
        return []

    # 查询规则详情
    conn = sqlite3.connect(DB_PATH)
    results = []
    for rid, words in triggered_rules.items():
        row = conn.execute(
            'SELECT level, category1, category2, standard, violation FROM rules WHERE rule_id=?',
            (rid,)
        ).fetchone()
        if not row:
            continue
        level, cat1, cat2, standard, violation = row
        category = " ".join(str(p) for p in [level, cat1, cat2] if p) or "未分类"
        results.append({
            "id": rid,
            "level": str(level or ""),
            "category": category,
            "standard": str(standard or ""),
            "violation": str(violation or ""),
            "penalty": "",
            "matched_words": words,
        })
    conn.close()

    return results


def check_message(message: str) -> dict:
    """
    检测销售话术是否违规（Embedding距离 + 超特异性关键词匹配）

    距离(distance)：Chroma cosine distance，范围[0,2]，0=完全相同，越大越不相关
    阈值(threshold)：距离≤阈值时判定命中

    评分策略：
    1. Embedding距离≤阈值：语义命中
    2. 超特异性关键词匹配（如"包过""挂靠"仅出现在1条规则中）：精准补充
    """
    # ── 阶段1：Embedding检索 ──────────────────────────────
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS,
        api_key=API_KEY, base_url=BASE_URL
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

    query = f"检索与销售话术违规相关的规则：{message}"
    emb_results = db.similarity_search_with_score(query, k=TOP_K)

    emb_hits = []
    for doc, distance in emb_results:
        distance = max(0.0, distance)
        rule_id = doc.metadata.get("id", 0)
        emb_hits.append({
            "id": rule_id,
            "level": doc.metadata.get("level", ""),
            "category": doc.metadata.get("category", ""),
            "standard": doc.metadata.get("standard", ""),
            "violation": doc.page_content,
            "penalty": doc.metadata.get("penalty", ""),
            "distance": round(distance, 4),
            "emb_distance": round(distance, 4),
            "keyword_score": 0.0,
            "hit_type": "语义匹配",
        })

    # ── 阶段2：违规触发词匹配 ──────────────────────────────
    kw_matches = _find_trigger_matches(message)

    # 关键词命中：如果embedding已召回同规则，跳过（避免重复）
    emb_rule_ids = {h["id"] for h in emb_hits}
    kw_supplement = []
    for kw in kw_matches:
        if kw["id"] not in emb_rule_ids:
            kw_supplement.append({
                "id": kw["id"],
                "level": kw["level"],
                "category": kw["category"],
                "standard": kw["standard"],
                "violation": kw["violation"],
                "penalty": kw["penalty"],
                "distance": DISTANCE_THRESHOLD,  # 虚拟距离=阈值
                "emb_distance": None,
                "keyword_score": 1.0,
                "hit_type": "触发词命中",
                "matched_keywords": kw["matched_words"],
            })

    # ── 阶段3：合并 + 筛选 ───────────────────────────────
    # 对embedding命中：如果有同规则的触发词命中，标注
    for hit in emb_hits:
        for kw in kw_matches:
            if hit["id"] == kw["id"]:
                hit["keyword_score"] = 1.0
                hit["hit_type"] = "语义+触发词"
                hit["matched_keywords"] = kw["matched_words"]
                hit["distance"] = round(max(0.0, hit["emb_distance"] - 0.15), 4)
                break

    combined = emb_hits + kw_supplement
    combined.sort(key=lambda x: x["distance"])

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
