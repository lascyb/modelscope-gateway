"""
ModelScopeClient 封装 ModelScope API 调用，集成负载均衡和使用量追踪。

使用场景：
- 自动选择可用模型进行 API 调用
- 智能路由：通过本地 AI 分析任务复杂度，选择合适的模型
- 从响应头自动获取配额限制
- 记录每次调用的使用量
- 支持 OpenAI 兼容的 API 格式
"""

import json
import os
from typing import Any, Dict, Generator, List, Optional, Union

import httpx

from .limits_tracker import LimitsTracker
from .load_balancer import LoadBalancer, ModelConfig
from .task_analyzer import TaskAnalyzer, AnalysisResult


class NoAvailableModelError(Exception):
    """NoAvailableModelError 当没有可用模型时抛出。"""

    pass


class RateLimitError(Exception):
    """RateLimitError 当所有模型都返回 429 时抛出。"""

    pass


class AuthenticationError(Exception):
    """AuthenticationError 当 API 密钥无效时抛出 (401)。"""

    pass


class ModelScopeClient:
    """ModelScopeClient 封装 ModelScope API 调用，支持负载均衡和智能路由。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: str = "models_config.json",
        usage_dir: str = "usage",
        limits_dir: str = "limits",
        base_url: str = "https://api-inference.modelscope.cn/v1",
    ):
        """
        ModelScopeClient 初始化客户端。

        Args:
            api_key: API 密钥，如果为 None 则从环境变量 MODELSCOPE_API_KEY 获取
            config_path: 模型配置文件路径
            usage_dir: 使用量记录存储目录
            limits_dir: 限制信息存储目录
            base_url: API 基础 URL
        """
        self._api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "API 密钥未提供，请设置 MODELSCOPE_API_KEY 环境变量或传入 api_key 参数"
            )

        self._base_url = base_url.rstrip("/")
        self._limits_tracker = LimitsTracker(limits_dir=limits_dir)
        self._balancer = LoadBalancer(
            config_path=config_path,
            limits_tracker=self._limits_tracker,
        )

        self._http_client = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

        # 初始化智能路由
        self._task_analyzer: Optional[TaskAnalyzer] = None
        self._smart_routing_enabled = False
        self._init_smart_routing()

    def _init_smart_routing(self) -> None:
        """_init_smart_routing 初始化智能路由。"""
        config = self._balancer.get_smart_routing_config()
        if not config.get("enabled", False):
            return

        local_ai_config = config.get("local_ai", {})
        base_url = local_ai_config.get("base_url", "http://localhost:11434")
        model = local_ai_config.get("model", "qwen2.5:1.5b")

        self._task_analyzer = TaskAnalyzer(base_url=base_url, model=model)

        # 检查本地 AI 是否可用
        if self._task_analyzer.is_available():
            self._smart_routing_enabled = True
            print(f"[智能路由] 已启用，本地模型: {model}")
        else:
            print(f"[智能路由] 本地 AI 服务不可用 ({base_url})，将使用默认路由")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model_id: Optional[str] = None,
        stream: bool = False,
        smart_route: bool = True,
        max_retries: int = 5,
        **kwargs,
    ) -> Union[Dict[str, Any], Generator]:
        """
        chat 发起聊天请求。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model_id: 指定模型 ID，如果为 None 则自动选择
            stream: 是否使用流式响应
            smart_route: 是否启用智能路由（仅当未指定 model_id 时生效）
            max_retries: 遇到 401/429 时最大重试次数（切换模型重试）
            **kwargs: 其他传递给 API 的参数

        Returns:
            API 响应结果

        Raises:
            NoAvailableModelError: 没有可用模型时抛出
            RateLimitError: 所有模型都返回 429 时抛出
            AuthenticationError: 所有模型都返回 401 时抛出
        """
        analysis_result: Optional[AnalysisResult] = None
        target_tier: Optional[int] = None
        tried_models: List[str] = []
        specified_model = model_id is not None

        # 智能路由：分析任务复杂度（只分析一次）
        if not specified_model and smart_route and self._smart_routing_enabled and self._task_analyzer:
            analysis_result = self._task_analyzer.analyze(messages)
            target_tier = analysis_result.suggested_model_tier
            print(
                f"[智能路由] 复杂度: {analysis_result.complexity.name} "
                f"(分数: {analysis_result.score}/10) -> 建议层级: {target_tier}"
            )

        for retry in range(max_retries + 1):
            # 选择模型
            if specified_model:
                model = self._get_model_by_id(model_id)
                if not model:
                    raise ValueError(f"指定的模型不存在: {model_id}")
            else:
                model = self._balancer.get_available_model(
                    target_tier=target_tier, exclude_models=tried_models
                )
                if not model:
                    if tried_models:
                        # 检查是 401 还是 429 导致的
                        raise NoAvailableModelError(
                            f"所有模型不可用，已尝试: {tried_models}"
                        )
                    raise NoAvailableModelError("没有可用的模型，所有模型配额已用尽")

            payload = {
                "model": model.id,
                "messages": messages,
                "stream": stream,
                **kwargs,
            }

            try:
                if stream:
                    try:
                        return self._stream_chat(
                            model.id, payload, analysis_result, tried_models
                        )
                    except (AuthenticationError, RateLimitError):
                        # 流式请求遇到 401/429，继续循环尝试下一个模型
                        if specified_model:
                            raise
                        continue

                response = self._http_client.post("/chat/completions", json=payload)

                # 处理 401 认证错误 - 切换模型重试
                if response.status_code == 401:
                    tried_models.append(model.id)
                    print(f"[401] 模型 {model.id} 认证错误，切换到下一个模型...")
                    if specified_model:
                        raise AuthenticationError(
                            f"模型 {model_id} 认证失败 (401)"
                        )
                    continue  # 重试下一个模型

                # 处理 429 错误 - 切换模型重试
                if response.status_code == 429:
                    self._handle_rate_limit(model.id, response, tried_models)
                    if specified_model:
                        raise RateLimitError(f"模型 {model_id} 配额已用尽 (429)")
                    continue  # 重试下一个模型

                response.raise_for_status()

                # 从响应头更新限制信息
                self._update_limits_from_response(model.id, response.headers)

                # 记录使用量
                self._balancer.record_usage(model.id)

                data = response.json()
                result = {
                    "model": model.id,
                    "content": data["choices"][0]["message"]["content"],
                    "usage": data.get("usage", {}),
                    "limits": self._get_current_limits(model.id),
                    "raw_response": data,
                }

                # 添加智能路由分析结果
                if analysis_result:
                    result["routing"] = {
                        "complexity": analysis_result.complexity.name,
                        "score": analysis_result.score,
                        "reason": analysis_result.reason,
                        "target_tier": target_tier,
                        "actual_tier": model.tier,
                    }

                # 如果有重试，记录
                if tried_models:
                    result["retried_models"] = tried_models

                return result

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    tried_models.append(model.id)
                    print(f"[401] 模型 {model.id} 认证错误，切换到下一个模型...")
                    if specified_model:
                        raise AuthenticationError(
                            f"模型 {model_id} 认证失败 (401)"
                        )
                    continue
                if e.response.status_code == 429:
                    self._handle_rate_limit(model.id, e.response, tried_models)
                    if specified_model:
                        raise RateLimitError(f"模型 {model_id} 配额已用尽 (429)")
                    continue
                raise

        # 所有重试都失败
        raise NoAvailableModelError(f"达到最大重试次数，已尝试模型: {tried_models}")

    def _handle_rate_limit(
        self,
        model_id: str,
        response: httpx.Response,
        tried_models: List[str],
    ) -> None:
        """
        _handle_rate_limit 处理 429 速率限制错误。

        Args:
            model_id: 模型 ID
            response: HTTP 响应
            tried_models: 已尝试的模型列表
        """
        # 尝试从响应头获取限制信息
        self._update_limits_from_response(model_id, response.headers)

        # 标记模型配额用尽
        self._limits_tracker.mark_model_exhausted(model_id)

        tried_models.append(model_id)
        print(f"[429] 模型 {model_id} 配额已用尽，切换到下一个模型...")

    def _stream_chat(
        self,
        model_id: str,
        payload: dict,
        analysis_result: Optional[AnalysisResult] = None,
        tried_models: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        _stream_chat 流式聊天实现。

        Args:
            model_id: 模型 ID
            payload: 请求负载
            analysis_result: 智能路由分析结果
            tried_models: 已尝试的模型列表

        Yields:
            流式响应块

        Raises:
            AuthenticationError: 401 错误，需要切换模型
            RateLimitError: 429 错误，需要切换模型
        """
        tried_models = tried_models or []

        with self._http_client.stream(
            "POST", "/chat/completions", json=payload
        ) as response:
            # 处理 401 认证错误 - 抛出让调用者切换模型
            if response.status_code == 401:
                tried_models.append(model_id)
                print(f"[401] 模型 {model_id} 认证错误，切换到下一个模型...")
                raise AuthenticationError(f"模型 {model_id} 认证失败 (401)")

            # 处理 429 错误
            if response.status_code == 429:
                self._handle_rate_limit(model_id, response, tried_models)
                raise RateLimitError(f"模型 {model_id} 配额已用尽 (429)")

            response.raise_for_status()
            # 从响应头更新限制信息
            self._update_limits_from_response(model_id, response.headers)

            # 记录使用量
            self._balancer.record_usage(model_id)

            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        done_chunk = {
                            "model": model_id,
                            "content": "",
                            "done": True,
                            "limits": self._get_current_limits(model_id),
                        }
                        if analysis_result:
                            model = self._get_model_by_id(model_id)
                            done_chunk["routing"] = {
                                "complexity": analysis_result.complexity.name,
                                "score": analysis_result.score,
                                "reason": analysis_result.reason,
                                "actual_tier": model.tier if model else None,
                            }
                        if tried_models:
                            done_chunk["retried_models"] = tried_models
                        yield done_chunk
                        break

                    try:
                        data = json.loads(data_str)
                        if data.get("choices") and data["choices"][0].get(
                            "delta", {}
                        ).get("content"):
                            yield {
                                "model": model_id,
                                "content": data["choices"][0]["delta"]["content"],
                                "done": False,
                            }
                    except json.JSONDecodeError:
                        continue

    def _update_limits_from_response(
        self, model_id: str, headers: httpx.Headers
    ) -> Dict[str, int]:
        """
        _update_limits_from_response 从响应头更新限制信息。

        Args:
            model_id: 模型 ID
            headers: 响应头

        Returns:
            解析后的限制信息
        """
        headers_dict = dict(headers)
        return self._limits_tracker.update_from_headers(model_id, headers_dict)

    def _get_current_limits(self, model_id: str) -> Dict[str, Optional[int]]:
        """
        _get_current_limits 获取当前模型的限制信息。

        Args:
            model_id: 模型 ID

        Returns:
            限制信息字典
        """
        return {
            "model_limit": self._limits_tracker.get_model_limit(model_id),
            "model_remaining": self._limits_tracker.get_model_remaining(model_id),
            "global_limit": self._limits_tracker.get_global_limit(),
            "global_remaining": self._limits_tracker.get_global_remaining(),
        }

    def _get_model_by_id(self, model_id: str) -> Optional[ModelConfig]:
        """_get_model_by_id 根据 ID 获取模型配置。"""
        for model in self._balancer.get_all_models():
            if model.id == model_id:
                return model
        return None

    def analyze_task(self, messages: List[Dict[str, str]]) -> Optional[AnalysisResult]:
        """
        analyze_task 分析任务复杂度（不发起实际请求）。

        Args:
            messages: 消息列表

        Returns:
            分析结果，如果智能路由未启用返回 None
        """
        if not self._smart_routing_enabled or not self._task_analyzer:
            return None
        return self._task_analyzer.analyze(messages)

    def get_status(self) -> dict:
        """
        get_status 获取当前使用状态。

        Returns:
            全局状态字典
        """
        status = self._balancer.get_global_status()
        status["smart_routing_available"] = self._smart_routing_enabled
        return status

    def get_available_model(
        self, target_tier: Optional[int] = None
    ) -> Optional[ModelConfig]:
        """
        get_available_model 获取当前可用的模型。

        Args:
            target_tier: 目标层级

        Returns:
            可用的模型配置，如果没有可用模型返回 None
        """
        return self._balancer.get_available_model(target_tier=target_tier)

    def reload_config(self) -> None:
        """reload_config 重新加载模型配置并重新初始化智能路由。"""
        self._balancer.reload_config()
        self._init_smart_routing()

    @property
    def smart_routing_enabled(self) -> bool:
        """smart_routing_enabled 智能路由是否启用。"""
        return self._smart_routing_enabled

    @property
    def balancer(self) -> LoadBalancer:
        """balancer 获取负载均衡器实例。"""
        return self._balancer

    @property
    def limits_tracker(self) -> LimitsTracker:
        """limits_tracker 获取限制追踪器实例。"""
        return self._limits_tracker

    @property
    def task_analyzer(self) -> Optional[TaskAnalyzer]:
        """task_analyzer 获取任务分析器实例。"""
        return self._task_analyzer

    def __del__(self):
        """__del__ 清理资源。"""
        if hasattr(self, "_http_client"):
            self._http_client.close()
