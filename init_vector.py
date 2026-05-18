# 初始化向量库脚本
# 功能：读取PDF文档，通过VLM识图提取结构化规则，向量化后存储到Chroma数据库
# 使用方法：
#   python init_vector.py --pdf "PDF文件路径"    # 自动：PDF→图片→VLM识图→入库
#   python init_vector.py --txt "TXT文件路径"    # 手动：直接加载文本文件入库

import argparse
import json
import os
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    BASE_URL, API_KEY, EMBEDDING_MODEL, VLM_MODEL,
    PERSIST_DIR, validate_config,
)

# ── VLM 识图提取 ──────────────────────────────────────────

VLM_PROMPT = """\
你是一个结构化数据提取专家。请仔细阅读这张表格图片，将每一行规则提取为严格的JSON数组格式。

每条规则包含以下字段：
- "level": 等级（如A类/B类/C类/D类）
- "category1": 一级类别
- "category2": 二级类别（如果没有则填null）
- "standard": 标准列内容（应该怎么做）
- "violation": 违规列内容（什么算违规）
- "penalty": 处理结果

要求：
1. 只输出JSON数组，不要输出任何其他文字或markdown标记
2. 表格中合并单元格的内容要为每行重复填写
3. 所有文字必须准确，不要编造或猜测
4. 如果某个单元格内容有多条（如①②③），合并为一条字符串"""


def _pdf_to_images(pdf_path: str) -> list[str]:
    """将PDF每页转为PNG图片，返回临时文件路径列表"""
    import fitz

    doc = fitz.open(pdf_path)
    paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        path = tempfile.mktemp(suffix=".png")
        pix.save(path)
        paths.append(path)
    doc.close()
    return paths


def _image_to_base64(image_path: str) -> str:
    """读取图片文件并编码为base64"""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_rules_with_vlm(image_path: str) -> list[dict]:
    """调用VLM视觉模型从图片中提取结构化规则数据（自动分块避免截断）"""
    from openai import OpenAI
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size

    # 估算规则密度：每500像素高度约10-15条规则，VLM输出上限约20条
    # 如果图片太高则切分
    MAX_CHUNK_HEIGHT = 1200
    if h <= MAX_CHUNK_HEIGHT:
        chunks = [(0, h)]
    else:
        chunks = []
        y = 0
        while y < h:
            end = min(y + MAX_CHUNK_HEIGHT, h)
            chunks.append((y, end))
            y = end

    all_rules = []
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for idx, (y_start, y_end) in enumerate(chunks):
        # 裁剪子图
        crop = img.crop((0, y_start, w, y_end))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            crop_path = tmp.name
        crop.save(crop_path)
        b64 = _image_to_base64(crop_path)
        os.remove(crop_path)

        # 为后续块添加上下文提示
        prompt = VLM_PROMPT
        if idx > 0:
            prompt += "\n\n注意：这是表格的下半部分，前半部分已提取完毕，只需提取本图中的规则。"

        try:
            response = client.chat.completions.create(
                model=VLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }],
                temperature=0.1,
                max_tokens=8192,
            )

            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            try:
                rules = json.loads(text)
            except json.JSONDecodeError:
                repaired = text
                while repaired.count("{") > repaired.count("}"):
                    repaired += "}"
                if repaired.count("[") > repaired.count("]"):
                    repaired += "]"
                try:
                    rules = json.loads(repaired)
                except json.JSONDecodeError:
                    print(f"[警告] 第 {idx+1} 块VLM返回的JSON无法解析")
                    continue

            all_rules.extend(rules)
            print(f"[INFO]   第 {idx+1}/{len(chunks)} 块提取到 {len(rules)} 条规则")

        except Exception as e:
            print(f"[警告] 第 {idx+1} 块VLM提取失败: {e}")

    return all_rules


def _rules_to_documents(rules: list[dict], source: str) -> list[Document]:
    """将VLM提取的结构化规则转为Document列表，同时写入业务数据库

    page_content 仅包含违规描述（用于embedding匹配），
    其他字段存入 metadata（用于结果展示）。
    """
    import sqlite3

    documents = []
    for i, rule in enumerate(rules):
        # 以违规描述作为主内容，标准关键词作为前缀增强匹配
        violation_text = rule.get("violation", "").strip()
        if not violation_text:
            continue

        standard_text = rule.get("standard", "").strip()
        if standard_text:
            content = f"{standard_text}：{violation_text}"
        else:
            content = violation_text

        # 构建分类标签
        parts = []
        if rule.get("level"):
            parts.append(rule["level"])
        if rule.get("category1"):
            parts.append(rule["category1"])
        if rule.get("category2"):
            parts.append(rule["category2"])
        label = " ".join(parts) if parts else "未分类"

        documents.append(Document(
            page_content=content,
            metadata={
                "source": source,
                "id": i + 1,
                "level": rule.get("level", ""),
                "category": label,
                "standard": rule.get("standard", ""),
                "penalty": rule.get("penalty", ""),
            }
        ))

    # 同步写入业务数据库
    _save_to_rules_db(rules, source)

    return documents


def _save_to_rules_db(rules: list[dict], source: str):
    """将结构化规则写入业务数据库 rules.db"""
    import sqlite3

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc", "rules.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        level TEXT,
        category1 TEXT,
        category2 TEXT,
        standard TEXT,
        violation TEXT,
        penalty TEXT,
        source TEXT,
        rule_id INTEGER,
        created_at TEXT DEFAULT (datetime('now','localtime'))
    )''')

    # 清空旧数据重新写入
    c.execute('DELETE FROM rules')

    for i, rule in enumerate(rules):
        c.execute('''INSERT INTO rules (level, category1, category2, standard, violation, penalty, source, rule_id)
                     VALUES (?,?,?,?,?,?,?,?)''',
                  (rule.get("level", ""),
                   rule.get("category1", ""),
                   rule.get("category2", ""),
                   rule.get("standard", ""),
                   rule.get("violation", ""),
                   rule.get("penalty", ""),
                   source,
                   i + 1))

    conn.commit()
    count = c.execute('SELECT COUNT(*) FROM rules').fetchone()[0]
    conn.close()
    print(f"[INFO] 业务数据库已同步: {db_path} ({count} 条规则)")

# ── TXT 文件加载 ──────────────────────────────────────────

def _load_txt(txt_path: str) -> list[Document]:
    """加载纯文本文件，按段落拆分为Document"""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    return [
        Document(page_content=p, metadata={"source": txt_path, "id": i + 1})
        for i, p in enumerate(paragraphs)
    ]

# ── 主流程 ────────────────────────────────────────────────

def init_vector_db(file_path: str):
    """
    初始化向量数据库

    参数:
        file_path: 文档路径（支持 PDF 和 TXT）

    处理流程:
        PDF: PDF→图片→VLM识图→结构化规则→入库
        TXT: 直接加载文本→入库
    """
    if not os.path.exists(file_path):
        print(f"[错误] 文件不存在: {file_path}")
        doc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc")
        if not os.path.exists(doc_dir):
            print(f"[提示] doc 目录不存在，请先创建并放入文件")
        else:
            print(f"[提示] 请将文件放入 doc/ 目录后重试")
        return

    ext = os.path.splitext(file_path)[1].lower()

    # ── 步骤1：加载文档 ────────────────────────────────
    if ext == ".txt":
        print(f"[INFO] 正在加载文本文件: {file_path}")
        docs = _load_txt(file_path)
    elif ext == ".pdf":
        print(f"[INFO] 正在加载PDF: {file_path}")
        print(f"[INFO] 步骤1/4: PDF转图片...")
        image_paths = _pdf_to_images(file_path)

        all_rules = []
        for idx, img_path in enumerate(image_paths):
            print(f"[INFO] 步骤2/4: VLM识图提取第 {idx+1}/{len(image_paths)} 页...")
            try:
                rules = _extract_rules_with_vlm(img_path)
                all_rules.extend(rules)
                print(f"[INFO]   第 {idx+1} 页提取到 {len(rules)} 条规则")
            except Exception as e:
                print(f"[警告]   第 {idx+1} 页VLM提取失败: {e}")
            finally:
                os.remove(img_path)

        if not all_rules:
            print("[错误] 未能从PDF中提取任何规则")
            return

        print(f"[INFO] VLM共提取 {len(all_rules)} 条规则")
        docs = _rules_to_documents(all_rules, file_path)
    else:
        print(f"[错误] 不支持的文件格式: {ext}，仅支持 PDF 和 TXT")
        return

    if not docs:
        print("[错误] 未能生成任何文档内容")
        return
    print(f"[INFO] 文档加载完成，共 {len(docs)} 条规则")

    # ── 步骤2：拆分文本 ────────────────────────────────
    print(f"[INFO] 步骤3/4: 拆分文本...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", " "],
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        print("[错误] 文本拆分结果为空")
        return
    print(f"[INFO] 文本拆分完成，共 {len(chunks)} 个文本块")

    # ── 步骤3：清空旧库 + 向量化入库 ──────────────────
    print(f"[INFO] 步骤4/4: 清空旧库并写入向量数据...")
    _reset_vector_db()

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        base_url=BASE_URL
    )

    BATCH_SIZE = 32
    db = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
        else:
            db.add_documents(batch)
        print(f"[INFO]   已处理 {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} 条")

    print(f"[SUCCESS] 向量库初始化完成！存储目录: {PERSIST_DIR}")
    print(f"[INFO] 共存储 {len(chunks)} 条向量记录")


def _reset_vector_db():
    """清空现有的向量数据库"""
    import chromadb
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        for c in client.list_collections():
            client.delete_collection(c.name)
    except Exception:
        pass

    import shutil
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)


if __name__ == "__main__":
    validate_config()

    parser = argparse.ArgumentParser(
        description="销售话术合规检测系统 - 向量库初始化工具"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdf",
        help="PDF文件路径（自动：PDF→图片→VLM识图→入库）"
    )
    input_group.add_argument(
        "--txt",
        help="TXT文本文件路径（手动：直接加载文本入库）"
    )
    args = parser.parse_args()

    init_vector_db(args.pdf or args.txt)
