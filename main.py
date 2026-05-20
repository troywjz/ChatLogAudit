# 销售话术合规检测系统 - 主程序
# 功能：基于Embedding距离 + 关键词匹配检测销售话术是否违规
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
    BASE_URL, API_KEY, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS,
    DISTANCE_THRESHOLD, TOP_K, PERSIST_DIR,
    validate_config,
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc", "rules.db")
CONFIG_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc", "config.db")


def _load_non_trigger_words() -> set[str]:
    """从config.db加载排除词表"""
    if not os.path.exists(CONFIG_DB_PATH):
        return set()
    conn = sqlite3.connect(CONFIG_DB_PATH)
    rows = conn.execute('SELECT word FROM non_trigger_words').fetchall()
    conn.close()
    return {r[0] for r in rows}


def _build_trigger_dict() -> dict[str, list[int]]:
    """从rules.db自动生成违规触发词字典

    从每条规则的violation文本中：
    1. 用jieba分词提取完整词语（避免跨词边界的噪声子串）
    2. 用标点分句提取2-8字完整短句片段（捕获"不用考试""7天无理由"等多字短语）

    保留仅出现在≤2条规则中且不在排除表中的词/短语作为触发词。
    规则库更新后自动生效，无需手动维护触发词。
    """
    if not os.path.exists(DB_PATH):
        return {}

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT rule_id, violation FROM rules'
    ).fetchall()
    conn.close()

    import jieba
    phrase_to_rules: dict[str, set[int]] = {}

    for rule_id, violation in rows:
        if not violation:
            continue

        # 来源1：jieba分词（2字以上的完整词）
        for word in jieba.cut(violation):
            word = word.strip()
            if len(word) >= 2 and not word.isspace():
                if word not in phrase_to_rules:
                    phrase_to_rules[word] = set()
                phrase_to_rules[word].add(rule_id)

        # 来源2：标点分句片段（2-8字的完整语义段）
        clauses = re.split(r'[，。；、：\s①②③④⑤⑥⑦⑧⑨⑩（）\(\)]+', violation)
        for clause in clauses:
            clause = clause.strip()
            if 2 <= len(clause) <= 8:
                # 提取连续中文字符+数字+字母的段
                seg = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z%]', '', clause)
                if 2 <= len(seg) <= 8 and seg not in phrase_to_rules:
                    if seg not in phrase_to_rules:
                        phrase_to_rules[seg] = set()
                    phrase_to_rules[seg].add(rule_id)

    # 过滤生成触发词字典
    non_trigger_words = _load_non_trigger_words()
    triggers: dict[str, list[int]] = {}
    for phrase, rule_ids in phrase_to_rules.items():
        if len(rule_ids) > 2:
            continue
        if phrase in non_trigger_words:
            continue
        if phrase.isdigit() or len(phrase) < 2:
            continue
        triggers[phrase] = sorted(rule_ids)

    return triggers


def _find_trigger_matches(message: str) -> list[dict]:
    """查找违规触发词匹配

    从rules.db自动生成触发词字典，对用户消息做子串匹配。
    匹配到的词在正常合规话术中几乎不会出现，因此可确信违规。
    """
    triggers = _build_trigger_dict()

    # 收集所有命中的触发词及对应规则ID
    triggered_rules: dict[int, list[str]] = {}
    for trigger, rule_ids in triggers.items():
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
    检测销售话术是否违规（Embedding距离 + 违规触发词匹配）

    距离(distance)：Chroma cosine distance，范围[0,2]，0=完全相同，越大越不相关
    阈值(threshold)：距离≤阈值时判定命中

    评分策略：
    1. Embedding距离≤阈值：语义命中
    2. 违规触发词匹配（从rules.db自动生成）：精准补充
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
                "distance": DISTANCE_THRESHOLD,
                "emb_distance": None,
                "keyword_score": 1.0,
                "hit_type": "触发词命中",
                "matched_keywords": kw["matched_words"],
            })

    # ── 阶段3：合并 + 筛选 ───────────────────────────────
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
