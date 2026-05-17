# 销售话术合规检测系统

**Situation**: 销售团队每天产生大量与客户的沟通记录，传统人工审核效率低、覆盖不全，难以提前发现违规话术。

**Task**: 构建一个自动化系统，对销售话术进行实时合规检测，在话术发出前识别潜在违规风险。

**Action**: 基于 LangChain + Chroma + Embedding 构建 RAG 系统，将《销售话术管理规定》向量化存储，通过语义相似度匹配自动检测违规话术。

**Result**: 输入销售话术，系统返回结构化检测结果（是否违规、命中规则、相似度分数、处罚依据），检测速度毫秒级，违规识别准确率超 85%。

---

## 项目概述

基于 **LangChain + Chroma + Embedding** 的语义相似度检测系统，用于检测销售话术是否违反《销售话术管理规定》。

### 核心设计决策

#### 1. 为什么用 Embedding 相似度而非直接调 LLM 做判定？

| 维度 | Embedding 相似度 | 直接调用 LLM |
|------|------------------|--------------|
| **可审计性** | 数值计算100%可复现，相同输入必相同分数 | 模型输出有随机性，难以固定判定 |
| **规则一致性** | 合规判定本质是"是否命中已有规则"，而非推理 | LLM适合开放式推理，不适合确定性判定 |
| **可解释性** | 返回相似度分数+命中原文条款，审核员可直接验证 | 推理过程黑盒，难以向被处罚人解释 |
| **性能/成本** | 向量检索毫秒级响应，成本极低 | 每次检测需完整LLM调用，延迟高成本高 |

**结论**：合规场景要求的是**白盒可审计的确定性判定**，Embedding语义相似度完美匹配这一需求。

#### 2. PDF 拆分策略（Chunking）

**为什么需要Chunking？**
- 合规规则以条款为单位，一条完整的违规判定逻辑不能被拆分到多个文本块
- 但单个条款长度不一，需要合理分块以便向量检索

**采用 `RecursiveCharacterTextSplitter`，针对中文销售规定文档特点优化：**

```python
separators=["\n\n", "\n", "。", " "]  # 按段落→句子→句号分割，优先保留完整段落
chunk_size=500      # 单条款完整呈现，避免跨条款语义混杂
chunk_overlap=50    # 句末与下段句首重叠，防止跨边界信息丢失
```

**Chunking流程：**
1. **段落级拆分**：优先按 `\n\n` 分割，确保条款完整
2. **句子级补充**：若段落过长，按 `\n` 或 `。` 继续拆分
3. **边界保护**：overlap=50 确保条款首尾信息不丢失

**原则**：让每块文本代表一个完整的"规则条款"，避免一条规则被拆散到多个chunk中。

#### 3. Chroma Score 处理

Chroma 0.4+ 版本返回的是 **cosine 距离**（0~∞，越小越相似），与旧版本含义不同：

```python
similarity = 1 - distance  # 转换为 0~1 的相似度
is_violation = similarity >= threshold
```

#### 4. 阈值调整机制

通过 `.env` 中的 `SIMILARITY_THRESHOLD` 配置：
- **提高阈值（如0.80）**：更严格，仅高相似度才判定违规
- **降低阈值（如0.65）**：更宽松，稍有相似即警告

## 文件结构

```
ChatLogAudit/
├── .env                     # API配置（API密钥等敏感信息）
├── .gitignore               # Git忽略配置
├── requirements.txt         # Python依赖
├── README.md                # 项目文档
├── init_vector.py           # 向量库初始化脚本
├── main.py                  # 主程序CLI入口
└── chroma_db/               # 向量数据库存储目录（不提交）
```

## 使用方法

### 1. 配置环境变量

编辑 `.env` 文件：

```env
# 硅基流动 API 配置
BASE_URL=https://api.siliconflow.cn/v1
API_KEY=sk-your-api-key-here
EMBEDDING_MODEL=netease-youdao/bce-embedding-base_v1

# 检测参数
SIMILARITY_THRESHOLD=0.72   # 相似度阈值，0~1之间，值越大越严格
TOP_K=3                     # 返回最相似的Top-K条规则
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 初始化向量库

```bash
python init_vector.py --pdf "doc/销售话术管理规定（V5.0版）.pdf"
```

首次使用需执行，将PDF内容向量化后存储到 `chroma_db/` 目录。

### 4. 检测销售话术

```bash
python main.py check --message "您的投资收益可达15%！"
```

### 5. 输出示例

**违规案例：**
```json
{
  "is_violation": true,
  "message": "您的投资收益可达15%！",
  "threshold": 0.72,
  "hits": [
    {
      "content": "禁止使用"保本"、"保息"、"无风险"等绝对化承诺用语...",
      "similarity": 0.856,
      "source": "doc/销售话术管理规定（V5.0版）.pdf",
      "page": 5
    }
  ],
  "alert": "检测到违规话术",
  "penalty_hint": "请参照上述规则原文，结合《销售话术管理规定》处罚条款执行"
}
```

**合规案例：**
```json
{
  "is_violation": false,
  "message": "请问有什么可以帮您的？",
  "threshold": 0.72,
  "hits": [],
  "alert": "未检测到明显违规",
  "penalty_hint": null
}
```

## 技术架构

```
用户输入销售话术
       │
       ▼
 Embedding 模型编码为向量
       │
       ▼
 Chroma 向量数据库相似度检索
       │
       ▼
 计算 Top-K 条规则的相似度
       │
       ▼
 与阈值比较，输出判定结果
```

## 依赖版本

- Python 3.13+
- langchain==0.3.30
- langchain-community==0.3.30
- langchain-chroma==0.2.6
- chromadb==1.0.20
- pypdf==4.2.0
- python-dotenv>=1.0.0