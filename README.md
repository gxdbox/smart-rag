# 🚀 Smart RAG

一个智能的 RAG（检索增强生成）问答系统，支持**混合检索**（向量检索 + BM25）和多种国内大模型。

## ✨ 核心特性

- 🔍 **混合检索**：向量检索（语义理解）+ BM25（精确匹配）+ 智能融合
- 🌐 **现代化 Web 界面**：基于 Streamlit 的直观交互体验
- 📄 **多格式支持**：支持 TXT、PDF、Markdown、Word、Excel 等格式
- 🏗️ **三层架构**：Retriever（检索）→ Ranker（排序）→ Generator（生成）
- 🤖 **国内大模型**：支持 DeepSeek、Moonshot、通义千问、智谱 GLM4 等
- 💾 **本地存储**：向量库 + BM25 索引，无需额外数据库
- 🧠 **知识图谱**：自动提取实体关系，增强检索效果
- 📊 **智能分块**：多种分块策略（固定长度、语义、段落、句子）

## 📦 安装

### 1. 克隆项目

```bash
git clone https://github.com/gxdbox/smart-rag.git
cd smart-rag
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## ⚙️ 配置

### 1. 创建环境变量文件

```bash
cp .env.example .env
```

### 2. 编辑 `.env` 文件

```env
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=你的API密钥
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=deepseek-chat
```

### 支持的国内大模型配置

#### DeepSeek

```env
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-xxx
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=deepseek-chat
```

#### Moonshot (Kimi)

```env
OPENAI_BASE_URL=https://api.moonshot.cn/v1
OPENAI_API_KEY=sk-xxx
EMBED_MODEL=moonshot-v1-8k
CHAT_MODEL=moonshot-v1-8k
```

#### 通义千问 (Qwen)

```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxx
EMBED_MODEL=text-embedding-v3
CHAT_MODEL=qwen-turbo
```

#### 智谱 GLM4

```env
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_API_KEY=xxx
EMBED_MODEL=embedding-3
CHAT_MODEL=glm-4-flash
```

> ⚠️ **注意**：不同模型提供商的 Embedding 模型名称可能不同，请参考各平台官方文档。

## 🚀 运行

### 本地运行

```bash
streamlit run app.py
```

默认访问地址：`http://localhost:8501`

### 指定端口运行

```bash
streamlit run app.py --server.port 8080
```

## 📖 使用说明

### 基础使用

1. **配置模型**：确保 `.env` 文件中已正确配置 API 信息
2. **上传文件**：在右侧区域上传文件（支持 TXT、PDF、MD、DOCX、XLSX 等）
3. **选择检索模式**：
   - **向量检索**：适合语义理解查询
   - **BM25 检索**：适合专业术语精确匹配
   - **混合检索**：综合两者优势（推荐）
4. **输入问题**：在问答区域输入您的问题
5. **生成回答**：点击"生成回答"按钮获取答案

### 高级功能

- **知识图谱**：自动提取文档中的实体关系
- **分块策略**：根据文档类型选择最优分块方式
- **索引同步**：自动检测并同步向量库和 BM25 索引
- **预处理数据导入**：批量导入已处理的文本数据

### 检索模式选择建议

| 查询类型 | 推荐模式 | 权重建议 |
|---------|---------|---------|
| 专业术语（如 "ABSD"） | BM25 或混合 | BM25 权重 0.7 |
| 语义理解（如 "如何理解架构设计"） | 向量或混合 | 向量权重 0.7 |
| 通用查询 | 混合检索 | 权重 0.5 |

## 🏗️ 项目结构

```
smart-rag/
├── app.py                      # Streamlit 主应用
├── rag_engine.py               # RAG 核心引擎
├── rag/                        # RAG 核心模块
│   ├── retriever/              # 检索器（向量、BM25、FAISS）
│   ├── ranker/                 # 排序器（相似度、重排序）
│   └── generator/              # 生成器（LLM）
├── file_utils.py               # 文件处理工具
├── chunk_strategy.py           # 分块策略
├── knowledge_graph.py          # 知识图谱
├── ocr_utils.py                # OCR 工具
├── docs/                       # 技术文档
│   ├── BM25_USAGE.md           # BM25 使用指南
│   ├── BM25_PRINCIPLE.md       # BM25 原理详解
│   ├── HYBRID_SEARCH_EXPLAINED.md  # 混合检索原理
│   └── UI_INTEGRATION_GUIDE.md     # UI 集成指南
├── vector_db.json              # 向量数据库（自动生成）
├── bm25_index.pkl              # BM25 索引（自动生成）
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── SECURITY.md                 # 安全说明
└── README.md                   # 项目说明
```

## 🔧 技术架构

### 三层架构设计

```
查询 → Retriever（检索） → Ranker（排序） → Generator（生成） → 答案
```

1. **Retriever 层**：
   - `VectorRetriever`：基于向量相似度的语义检索
   - `BM25Retriever`：基于 BM25 算法的关键词检索
   - `FAISSRetriever`：基于 FAISS 的高效向量检索（计划中）

2. **Ranker 层**：
   - `SimilarityRanker`：基于余弦相似度的排序
   - `RerankRanker`：基于重排序模型的精细排序（计划中）

3. **Generator 层**：
   - `LLMGenerator`：基于大语言模型的答案生成

### 混合检索原理

混合检索通过加权融合向量检索和 BM25 检索的结果：

```python
综合分数 = α × 向量分数 + (1-α) × BM25分数
```

- **向量检索**：擅长语义理解、同义词识别
- **BM25 检索**：擅长专业术语、精确匹配
- **混合检索**：综合两者优势，提供最佳检索效果

详细原理请参考：[`docs/HYBRID_SEARCH_EXPLAINED.md`](docs/HYBRID_SEARCH_EXPLAINED.md)

## 📚 文档

- [BM25 使用指南](docs/BM25_USAGE.md)
- [BM25 原理详解](docs/BM25_PRINCIPLE.md)
- [混合检索原理](docs/HYBRID_SEARCH_EXPLAINED.md)
- [UI 集成指南](docs/UI_INTEGRATION_GUIDE.md)
- [安全说明](SECURITY.md)
- [架构重构文档](REFACTORING.md)

## ☁️ 在线部署

### Streamlit Cloud

1. 将项目推送到 GitHub
2. 访问 [Streamlit Cloud](https://streamlit.io/cloud)
3. 连接 GitHub 仓库
4. 在 Secrets 中配置环境变量：
   ```toml
   OPENAI_BASE_URL = "https://api.deepseek.com"
   OPENAI_API_KEY = "你的密钥"
   EMBED_MODEL = "text-embedding-3-small"
   CHAT_MODEL = "deepseek-chat"
   ```

### Docker 部署

创建 `Dockerfile`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

构建并运行：

```bash
docker build -t smart-rag .
docker run -p 8501:8501 --env-file .env smart-rag
```

### 本地服务器部署

使用 `screen` 或 `tmux` 保持后台运行：

```bash
screen -S rag
streamlit run app.py --server.port 8501
# Ctrl+A+D 退出 screen
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
