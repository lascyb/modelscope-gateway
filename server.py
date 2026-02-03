"""
ModelScope API HTTP 服务 - 供 n8n 等外部系统调用。

启动方式：
    1. 复制 env.example 为 .env 并填入配置
    2. python server.py
    或
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

API 文档：
    http://localhost:8000/docs
"""

import os
from pathlib import Path
from typing import List, Optional

# 加载 .env 文件
from dotenv import load_dotenv

# 从当前目录或上级目录加载 .env
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core.api_client import (
    ModelScopeClient,
    NoAvailableModelError,
    RateLimitError,
    AuthenticationError,
)

# ============================================================
# 配置（从 .env 文件或环境变量读取）
# ============================================================

API_KEY = os.getenv("MODELSCOPE_API_KEY") or ""
HOST = os.getenv("SERVER_HOST") or "0.0.0.0"
PORT = int(os.getenv("SERVER_PORT") or "8000")

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="ModelScope API 网关",
    description="ModelScope API 负载均衡网关，支持智能路由、自动切换模型",
    version="1.0.0",
)

# CORS 配置（允许 n8n 等外部系统调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客户端实例
_client: Optional[ModelScopeClient] = None


def get_client() -> ModelScopeClient:
    """get_client 获取或创建客户端实例。"""
    global _client
    if _client is None:
        if not API_KEY:
            raise HTTPException(
                status_code=500,
                detail="MODELSCOPE_API_KEY 环境变量未设置",
            )
        _client = ModelScopeClient(api_key=API_KEY)
    return _client


# ============================================================
# 请求/响应模型
# ============================================================


class Message(BaseModel):
    """Message 聊天消息。"""

    role: str = Field(..., description="角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """ChatRequest 聊天请求。"""

    messages: List[Message] = Field(..., description="消息列表")
    model_id: Optional[str] = Field(None, description="指定模型ID，不指定则自动选择")
    stream: bool = Field(False, description="是否使用流式响应")
    smart_route: bool = Field(True, description="是否启用智能路由")
    max_tokens: Optional[int] = Field(None, description="最大生成token数")
    temperature: Optional[float] = Field(None, description="温度参数")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "system", "content": "你是一个有帮助的助手"},
                    {"role": "user", "content": "你好，请介绍一下你自己"},
                ],
                "stream": False,
                "smart_route": True,
                "max_tokens": 500,
            }
        }
    )


class ChatResponse(BaseModel):
    """ChatResponse 聊天响应。"""

    model: str = Field(..., description="使用的模型ID")
    content: str = Field(..., description="回复内容")
    usage: dict = Field(default_factory=dict, description="Token使用量")
    limits: dict = Field(default_factory=dict, description="配额限制信息")
    routing: Optional[dict] = Field(None, description="智能路由信息")
    retried_models: Optional[List[str]] = Field(None, description="跳过的模型列表")


class ModelInfo(BaseModel):
    """ModelInfo 模型信息。"""

    model_id: str
    name: str
    tier: int
    priority: int
    enabled: bool
    available: bool
    daily_limit: Optional[int]
    usage: int
    remaining: Optional[int]


class StatusResponse(BaseModel):
    """StatusResponse 状态响应。"""

    global_daily_limit: Optional[int]
    total_usage: int
    remaining: Optional[int]
    smart_routing_enabled: bool
    smart_routing_available: bool
    models: List[ModelInfo]


class ErrorResponse(BaseModel):
    """ErrorResponse 错误响应。"""

    error: str
    detail: str


# ============================================================
# OpenAI 兼容格式
# ============================================================


class OpenAIMessage(BaseModel):
    """OpenAIMessage OpenAI 格式消息。"""

    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    """OpenAIChatRequest OpenAI 兼容的聊天请求。"""

    model: Optional[str] = Field(None, description="模型ID，可选")
    messages: List[OpenAIMessage]
    temperature: Optional[float] = Field(None)
    max_tokens: Optional[int] = Field(None)
    stream: bool = Field(False)
    top_p: Optional[float] = Field(None)
    frequency_penalty: Optional[float] = Field(None)
    presence_penalty: Optional[float] = Field(None)
    stop: Optional[List[str]] = Field(None)


class OpenAIUsage(BaseModel):
    """OpenAIUsage Token 使用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChoiceMessage(BaseModel):
    """OpenAIChoiceMessage 回复消息。"""

    role: str = "assistant"
    content: str


class OpenAIChoice(BaseModel):
    """OpenAIChoice 选择项。"""

    index: int = 0
    message: OpenAIChoiceMessage
    finish_reason: str = "stop"


class OpenAIDelta(BaseModel):
    """OpenAIDelta 流式增量。"""

    role: Optional[str] = None
    content: Optional[str] = None


class OpenAIStreamChoice(BaseModel):
    """OpenAIStreamChoice 流式选择项。"""

    index: int = 0
    delta: OpenAIDelta
    finish_reason: Optional[str] = None


class OpenAIChatResponse(BaseModel):
    """OpenAIChatResponse OpenAI 兼容的聊天响应。"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


class OpenAIStreamResponse(BaseModel):
    """OpenAIStreamResponse OpenAI 兼容的流式响应。"""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[OpenAIStreamChoice]


class OpenAIErrorResponse(BaseModel):
    """OpenAIErrorResponse OpenAI 兼容的错误响应。"""

    error: dict


# ============================================================
# API 接口
# ============================================================


@app.get("/", summary="健康检查")
async def health_check():
    """健康检查接口。"""
    return {"status": "ok", "service": "ModelScope API Gateway"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        429: {"model": ErrorResponse, "description": "所有模型配额已用尽"},
        401: {"model": ErrorResponse, "description": "认证失败"},
        503: {"model": ErrorResponse, "description": "没有可用模型"},
    },
    summary="聊天接口",
)
async def chat(request: ChatRequest):
    """
    发起聊天请求。

    - 自动根据配置选择可用模型
    - 支持智能路由（根据任务复杂度选择模型）
    - 遇到 401/429 自动切换模型重试
    """
    client = get_client()

    # 构建消息列表
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # 构建额外参数
    kwargs = {}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature

    try:
        # 流式响应
        if request.stream:
            return StreamingResponse(
                _stream_chat(client, messages, request, kwargs),
                media_type="text/event-stream",
            )

        # 非流式响应
        response = client.chat(
            messages=messages,
            model_id=request.model_id,
            stream=False,
            smart_route=request.smart_route,
            **kwargs,
        )

        return ChatResponse(
            model=response["model"],
            content=response["content"],
            usage=response.get("usage", {}),
            limits=response.get("limits", {}),
            routing=response.get("routing"),
            retried_models=response.get("retried_models"),
        )

    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except NoAvailableModelError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_chat(client, messages, request, kwargs):
    """_stream_chat 流式聊天生成器。"""
    try:
        stream = client.chat(
            messages=messages,
            model_id=request.model_id,
            stream=True,
            smart_route=request.smart_route,
            **kwargs,
        )

        for chunk in stream:
            # SSE 格式
            import json

            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    except Exception as e:
        import json

        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"


@app.get(
    "/status",
    response_model=StatusResponse,
    summary="获取使用状态",
)
async def get_status():
    """
    获取当前使用状态。

    返回全局配额、各模型使用量和剩余配额。
    """
    client = get_client()
    status = client.get_status()

    models = []
    for m in status.get("models", []):
        models.append(
            ModelInfo(
                model_id=m["model_id"],
                name=m["name"],
                tier=m["tier"],
                priority=m["priority"],
                enabled=m["enabled"],
                available=m["available"],
                daily_limit=m.get("daily_limit"),
                usage=m["usage"],
                remaining=m.get("remaining"),
            )
        )

    return StatusResponse(
        global_daily_limit=status.get("global_daily_limit"),
        total_usage=status["total_usage"],
        remaining=status.get("remaining"),
        smart_routing_enabled=status.get("smart_routing_enabled", False),
        smart_routing_available=status.get("smart_routing_available", False),
        models=models,
    )


@app.get("/models", summary="获取模型列表")
async def get_models():
    """获取所有配置的模型列表。"""
    client = get_client()
    models = client.balancer.get_all_models()

    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "tier": m.tier,
                "enabled": m.enabled,
            }
            for m in models
        ]
    }


@app.post("/reload", summary="重新加载配置")
async def reload_config():
    """重新加载模型配置。"""
    client = get_client()
    client.reload_config()
    return {"status": "ok", "message": "配置已重新加载"}


# ============================================================
# OpenAI 兼容接口 (/v1/*)
# ============================================================


@app.post(
    "/v1/chat/completions",
    summary="OpenAI 兼容聊天接口",
    tags=["OpenAI"],
)
async def openai_chat_completions(request: OpenAIChatRequest):
    """
    OpenAI 兼容的聊天补全接口。

    兼容 OpenAI API 格式，可直接用于：
    - Cursor
    - Continue
    - OpenAI SDK
    - 其他兼容 OpenAI 的工具
    """
    import time
    import uuid
    import json

    client = get_client()

    # 构建消息
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # 构建额外参数
    kwargs = {}
    if request.max_tokens:
        kwargs["max_tokens"] = request.max_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stop:
        kwargs["stop"] = request.stop

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        # 流式响应
        if request.stream:
            return StreamingResponse(
                _openai_stream(client, messages, request, kwargs, request_id, created),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 非流式响应
        response = client.chat(
            messages=messages,
            model_id=request.model,
            stream=False,
            smart_route=True,  # 默认启用智能路由
            **kwargs,
        )

        # 构建 OpenAI 格式响应
        usage = response.get("usage", {})
        return OpenAIChatResponse(
            id=request_id,
            created=created,
            model=response["model"],
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChoiceMessage(
                        role="assistant",
                        content=response["content"],
                    ),
                    finish_reason="stop",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    except AuthenticationError as e:
        return _openai_error(401, "invalid_api_key", str(e))
    except RateLimitError as e:
        return _openai_error(429, "rate_limit_exceeded", str(e))
    except NoAvailableModelError as e:
        return _openai_error(503, "model_not_available", str(e))
    except Exception as e:
        return _openai_error(500, "internal_error", str(e))


async def _openai_stream(client, messages, request, kwargs, request_id, created):
    """_openai_stream OpenAI 格式流式生成器。"""
    import json

    model_name = request.model or "auto"

    try:
        stream = client.chat(
            messages=messages,
            model_id=request.model,
            stream=True,
            smart_route=True,
            **kwargs,
        )

        # 发送第一个 chunk（包含 role）
        first_chunk = OpenAIStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(role="assistant", content=""),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # 流式输出内容
        for chunk in stream:
            if "content" in chunk:
                content = chunk["content"]
                # 更新实际使用的模型名
                if "model" in chunk:
                    model_name = chunk["model"]

                stream_chunk = OpenAIStreamResponse(
                    id=request_id,
                    created=created,
                    model=model_name,
                    choices=[
                        OpenAIStreamChoice(
                            index=0,
                            delta=OpenAIDelta(content=content),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"

        # 发送结束标记
        final_chunk = OpenAIStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                OpenAIStreamChoice(
                    index=0,
                    delta=OpenAIDelta(),
                    finish_reason="stop",
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"error": {"message": str(e), "type": "server_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


def _openai_error(status_code: int, error_type: str, message: str):
    """_openai_error 返回 OpenAI 格式错误。"""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": error_type,
            }
        },
    )


@app.get(
    "/v1/models",
    summary="OpenAI 兼容模型列表",
    tags=["OpenAI"],
)
async def openai_list_models():
    """
    OpenAI 兼容的模型列表接口。

    返回格式与 OpenAI /v1/models 一致。
    """
    import time

    client = get_client()
    models = client.balancer.get_all_models()

    return {
        "object": "list",
        "data": [
            {
                "id": m.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "modelscope",
                "permission": [],
                "root": m.id,
                "parent": None,
            }
            for m in models
            if m.enabled
        ],
    }


@app.get(
    "/v1/models/{model_id:path}",
    summary="获取模型详情",
    tags=["OpenAI"],
)
async def openai_get_model(model_id: str):
    """获取指定模型的详情。"""
    import time

    client = get_client()
    models = client.balancer.get_all_models()

    for m in models:
        if m.id == model_id:
            return {
                "id": m.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "modelscope",
                "permission": [],
                "root": m.id,
                "parent": None,
            }

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


# ============================================================
# n8n 专用接口（简化版）
# ============================================================


class N8nChatRequest(BaseModel):
    """N8nChatRequest n8n 简化请求格式。"""

    prompt: str = Field(..., description="用户输入")
    system: Optional[str] = Field(None, description="系统提示词")
    model_id: Optional[str] = Field(None, description="指定模型ID")
    max_tokens: int = Field(1000, description="最大生成token数")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "你好，请帮我写一首诗",
                "system": "你是一个诗人",
                "max_tokens": 500,
            }
        }
    )


class N8nChatResponse(BaseModel):
    """N8nChatResponse n8n 简化响应格式。"""

    text: str = Field(..., description="回复内容")
    model: str = Field(..., description="使用的模型")


@app.post(
    "/n8n/chat",
    response_model=N8nChatResponse,
    summary="n8n 专用聊天接口",
    tags=["n8n"],
)
async def n8n_chat(request: N8nChatRequest):
    """
    n8n 专用简化接口。

    只需传入 prompt，自动处理消息格式。
    """
    client = get_client()

    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    try:
        response = client.chat(
            messages=messages,
            model_id=request.model_id,
            stream=False,
            max_tokens=request.max_tokens,
        )

        return N8nChatResponse(
            text=response["content"],
            model=response["model"],
        )

    except (AuthenticationError, RateLimitError, NoAvailableModelError) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print(f"启动服务: http://{HOST}:{PORT}")
    print(f"API 文档: http://{HOST}:{PORT}/docs")
    uvicorn.run(app, host=HOST, port=PORT)

