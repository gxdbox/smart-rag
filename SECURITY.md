# 🔒 安全说明

## ⚠️ 重要提醒

本项目需要配置 API Keys 才能运行。**请勿将真实的 API Keys 提交到 Git 仓库！**

---

## 🔑 API Keys 配置

### 1. 复制环境变量模板

```bash
cp .env.example .env
```

### 2. 编辑 `.env` 文件

在 `.env` 文件中填入你的真实 API Keys：

```env
# Embedding API（使用硅基流动）
EMBED_BASE_URL=https://api.siliconflow.cn/v1
EMBED_API_KEY=你的真实API密钥
EMBED_MODEL=BAAI/bge-m3

# Chat API（使用 DeepSeek）
CHAT_BASE_URL=https://api.deepseek.com
CHAT_API_KEY=你的真实API密钥
CHAT_MODEL=deepseek-chat
```

### 3. 验证 `.env` 已被忽略

确保 `.env` 文件在 `.gitignore` 中：

```bash
git check-ignore .env
# 应该输出: .env
```

---

## 🛡️ 安全检查清单

在推送代码前，请确认：

- [ ] `.env` 文件已添加到 `.gitignore`
- [ ] 代码中没有硬编码的 API Keys
- [ ] `.env.example` 只包含示例值，不包含真实密钥
- [ ] 运行 `git status` 确认 `.env` 不在待提交列表中

---

## 🔍 检查敏感数据

运行以下命令检查是否有敏感数据：

```bash
# 检查是否有 API Keys
grep -r "sk-" --exclude-dir=venv --exclude-dir=.git .

# 检查 git 状态
git status

# 查看将要提交的文件
git diff --cached
```

---

## 🚨 如果不小心提交了密钥

### 1. 立即撤销密钥

- 前往 API 提供商平台撤销泄露的密钥
- 生成新的密钥

### 2. 从 Git 历史中删除

```bash
# 从 Git 历史中完全删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送
git push origin --force --all
```

### 3. 使用 BFG Repo-Cleaner（推荐）

```bash
# 安装 BFG
brew install bfg

# 删除敏感文件
bfg --delete-files .env

# 清理
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 强制推送
git push origin --force --all
```

---

## 📝 最佳实践

1. **永远不要提交 `.env` 文件**
2. **使用 `.env.example` 作为模板**
3. **定期轮换 API Keys**
4. **使用环境变量管理工具**（如 direnv、dotenv）
5. **在 CI/CD 中使用 Secrets 管理**

---

## 🔗 相关资源

- [GitHub Secrets 文档](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [环境变量最佳实践](https://12factor.net/config)
- [Git Secrets 工具](https://github.com/awslabs/git-secrets)

---

**记住：安全第一！保护好你的 API Keys！** 🔐
