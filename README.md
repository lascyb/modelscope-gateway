# ModelScope API Gateway

ModelScope 魔搭社区 API 负载均衡网关，支持多模型自动切换、智能路由、配额管理。

## ✨ 特性

- **负载均衡** - 按优先级自动选择可用模型，单个模型配额用尽自动切换
- **智能路由** - 通过本地 AI 分析任务复杂度，自动分配到合适的模型层级
- **配额追踪** - 自动从 API 响应头获取配额信息，记录每日使用量
- **错误重试** - 遇到 401/429 错误自动切换模型重试
- **OpenAI 兼容** - 提供 `/v1/chat/completions` 接口，兼容 OpenAI SDK
- **Docker 部署** - 提供完整的 Docker 和 Docker Compose 配置

## 📦 安装

### 本地安装

```bash
# 克隆项目
git clone https://github.com/lascyb/modelscope.git
cd modelscope

# 创建虚拟环境（推荐）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp env.example .env
# 编辑 .env 填入 MODELSCOPE_API_KEY
```

### Docker 部署

```bash
# 配置环境变量
cp env.example .env
# 编辑 .env 填入 MODELSCOPE_API_KEY

# 启动服务
docker-compose --env-file .env -f deploy/docker-compose.yml up -d

# 查看日志
docker-compose -f deploy/docker-compose.yml logs -f
```

## 🚀 启动

### 命令行启动

```bash
python server.py
```

或使用 uvicorn：

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 访问 API 文档

启动后访问：http://localhost:8000/docs

## 📖 API 接口

### OpenAI 兼容接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全（支持流式） |
| `/v1/models` | GET | 模型列表 |
| `/v1/models/{model_id}` | GET | 模型详情 |

**使用 OpenAI SDK：**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any"  # 本地服务可填任意值
)

response = client.chat.completions.create(
    model="auto",  # 自动选择模型
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

**使用 curl：**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

### 原生接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat` | POST | 聊天接口（更多控制选项） |
| `/status` | GET | 使用状态和配额信息 |
| `/models` | GET | 模型列表 |
| `/reload` | POST | 重新加载配置 |
| `/n8n/chat` | POST | n8n 专用简化接口 |

**聊天请求示例：**

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "介绍一下你自己"}
  ],
  "stream": false,
  "smart_route": true,
  "max_tokens": 500
}
```

## ⚙️ 配置

### 环境变量 (.env)

```env
# ModelScope API 密钥（必填）
MODELSCOPE_API_KEY=your-api-key-here

# 服务器配置
SERVER_HOST=0.0.0.0
SERVER_PORT=8000

# 本地 AI 配置（智能路由，可选）
LOCAL_AI_BASE_URL=http://localhost:11434
LOCAL_AI_MODEL=qwen2.5:1.5b
```

### 模型配置 (models_config.json)

```json
{
  "models": [
    {
      "id": "deepseek-ai/DeepSeek-R1-0528",
      "name": "DeepSeek-R1",
      "tier": 1,
      "enabled": true
    },
    {
      "id": "Qwen/Qwen3-235B-A22B-Thinking-2507",
      "name": "Qwen3-235B",
      "tier": 2,
      "enabled": true
    }
  ],
  "smart_routing": {
    "enabled": true,
    "local_ai": {
      "base_url": "http://localhost:11434",
      "model": "qwen2.5:1.5b"
    }
  }
}
```

**配置说明：**

- `id` - ModelScope 模型 ID
- `name` - 显示名称
- `tier` - 模型层级（1=最强，4=最轻量）
- `enabled` - 是否启用
- 模型顺序决定优先级（越靠前优先级越高）

## 🧠 智能路由

智能路由使用本地 AI（如 Ollama）分析任务复杂度，自动选择合适的模型：

| 复杂度 | 分数 | 模型层级 | 示例任务 |
|--------|------|----------|----------|
| 简单 | 1-3 | Tier 4 | 打招呼、简单问答 |
| 中等 | 4-5 | Tier 3 | 解释概念、摘要 |
| 复杂 | 6-7 | Tier 2 | 代码编写、分析 |
| 非常复杂 | 8-10 | Tier 1 | 数学推理、系统设计 |

### 启用智能路由

1. 安装 Ollama: https://ollama.com
2. 下载模型：`ollama pull qwen2.5:1.5b`
3. 确保 `models_config.json` 中 `smart_routing.enabled` 为 `true`

## 📁 目录结构

```
modelscope/
├── core/                        # 核心模块
│   ├── __init__.py              # 包初始化
│   ├── api_client.py            # API 客户端
│   ├── load_balancer.py         # 负载均衡器
│   ├── limits_tracker.py        # 配额追踪器
│   ├── usage_tracker.py         # 使用量追踪器
│   └── task_analyzer.py         # 任务分析器
├── deploy/                      # 部署配置
│   ├── Dockerfile
│   ├── docker-compose.yml       # 生产环境
│   └── docker-compose.dev.yml   # 开发环境
├── usage/                       # 使用量记录 (YYYY-MM-DD.json)
├── limits/                      # 配额记录 (YYYY-MM-DD.json)
├── server.py                    # HTTP 服务入口
├── main.py                      # 命令行示例
├── models_config.json           # 模型配置
├── env.example                  # 环境变量模板
└── requirements.txt             # Python 依赖
```

## 🔧 Python SDK 使用

```python
from core import ModelScopeClient

# 创建客户端
client = ModelScopeClient(api_key="your-api-key")

# 发送聊天请求
response = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    smart_route=True,  # 启用智能路由
)

print(f"模型: {response['model']}")
print(f"回复: {response['content']}")

# 获取状态
status = client.get_status()
print(f"今日使用: {status['total_usage']}")
print(f"剩余配额: {status['remaining']}")
```

## 📊 配额说明

ModelScope 魔搭社区提供：

- **全局限制**: 每人每天 2000 次 API 调用
- **模型限制**: 每个模型有各自的每日限制

本网关自动从 API 响应头获取配额信息：

| 响应头 | 说明 |
|--------|------|
| `modelscope-ratelimit-requests-limit` | 用户当天限额 |
| `modelscope-ratelimit-requests-remaining` | 用户当天剩余 |
| `modelscope-ratelimit-model-requests-limit` | 模型当天限额 |
| `modelscope-ratelimit-model-requests-remaining` | 模型当天剩余 |

## 🐳 Docker

### 生产环境

```bash
docker-compose --env-file .env -f deploy/docker-compose.yml up -d
```

### 开发环境（热重载）

```bash
docker-compose --env-file .env -f deploy/docker-compose.dev.yml up
```

### 启用 Ollama（智能路由）

```bash
docker-compose --env-file .env -f deploy/docker-compose.yml --profile with-ollama up -d
```

## 📝 License

[Apache License 2.0](LICENSE)
