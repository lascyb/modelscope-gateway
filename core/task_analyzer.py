"""
TaskAnalyzer 使用本地 AI 分析任务复杂度，决定由哪个远程模型执行。

使用场景：
- 简单任务路由到小模型（节省配额、响应快）
- 复杂任务路由到大模型（能力强、效果好）
- 支持 Ollama 等本地 AI 服务
"""

import httpx
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, List, Dict, Any


class Complexity(IntEnum):
    """Complexity 任务复杂度等级。{1:简单, 2:中等, 3:复杂, 4:非常复杂}"""

    SIMPLE = 1  # 简单：打招呼、简单问答、格式转换
    MEDIUM = 2  # 中等：解释概念、简单代码、摘要
    COMPLEX = 3  # 复杂：代码编写、分析推理、长文创作
    VERY_COMPLEX = 4  # 非常复杂：多步推理、复杂代码、数学证明


@dataclass
class AnalysisResult:
    """AnalysisResult 任务分析结果。"""

    complexity: Complexity  # 复杂度等级
    score: int  # 复杂度分数 (1-10)
    reason: str  # 分析原因
    suggested_model_tier: int  # 建议的模型层级


ANALYSIS_PROMPT = """你是一个任务复杂度分析专家。分析用户的请求，评估其复杂度。

评估维度：
1. 知识深度：需要多少专业知识
2. 推理步骤：需要多少推理步骤
3. 创造性：需要多少创造力
4. 上下文：需要理解多少上下文
5. 输出长度：预期输出的长度

复杂度等级：
- 1 (简单): 打招呼、简单问答、格式转换、翻译短句
- 2 (中等): 解释概念、简单代码修改、文本摘要、翻译段落
- 3 (复杂): 代码编写、分析推理、长文创作、技术问题解决
- 4 (非常复杂): 多步数学推理、复杂系统设计、算法实现、学术论文

请直接返回一个 JSON 对象（不要 markdown 代码块），格式：
{"score": 1-10的分数, "level": 1-4的等级, "reason": "简短原因"}

用户请求：
{user_message}"""


class TaskAnalyzer:
    """TaskAnalyzer 使用本地 AI 分析任务复杂度。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        timeout: float = 30.0,
    ):
        """
        TaskAnalyzer 初始化分析器。

        Args:
            base_url: 本地 AI 服务地址（默认 Ollama）
            model: 用于分析的本地模型
            timeout: 请求超时时间
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def analyze(self, messages: List[Dict[str, str]]) -> AnalysisResult:
        """
        analyze 分析任务复杂度。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]

        Returns:
            分析结果
        """
        # 提取用户消息
        user_message = self._extract_user_message(messages)

        try:
            # 调用本地 AI 分析
            result = self._call_local_ai(user_message)
            return result
        except Exception as e:
            # 分析失败时返回默认中等复杂度
            return AnalysisResult(
                complexity=Complexity.MEDIUM,
                score=5,
                reason=f"分析失败，使用默认值: {str(e)}",
                suggested_model_tier=2,
            )

    def _extract_user_message(self, messages: List[Dict[str, str]]) -> str:
        """_extract_user_message 从消息列表提取用户消息。"""
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        return "\n".join(user_messages) if user_messages else ""

    def _call_local_ai(self, user_message: str) -> AnalysisResult:
        """_call_local_ai 调用本地 AI 进行分析。"""
        prompt = ANALYSIS_PROMPT.format(user_message=user_message)

        response = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # 低温度，更确定性
                    "num_predict": 200,  # 限制输出长度
                },
            },
        )
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data.get("response", ""))

    def _parse_response(self, response_text: str) -> AnalysisResult:
        """_parse_response 解析本地 AI 的响应。"""
        import json
        import re

        # 尝试提取 JSON
        try:
            # 尝试直接解析
            result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            json_match = re.search(r"\{[^}]+\}", response_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # 解析失败，返回默认值
                return AnalysisResult(
                    complexity=Complexity.MEDIUM,
                    score=5,
                    reason="无法解析分析结果",
                    suggested_model_tier=2,
                )

        score = int(result.get("score", 5))
        level = int(result.get("level", 2))
        reason = result.get("reason", "")

        # 确保值在有效范围内
        score = max(1, min(10, score))
        level = max(1, min(4, level))

        return AnalysisResult(
            complexity=Complexity(level),
            score=score,
            reason=reason,
            suggested_model_tier=self._score_to_tier(score),
        )

    def _score_to_tier(self, score: int) -> int:
        """
        _score_to_tier 将分数转换为模型层级。

        Args:
            score: 复杂度分数 (1-10)

        Returns:
            模型层级 (1=最强模型, 2=次强, ...)
        """
        if score <= 3:
            return 4  # 简单任务用最小模型
        elif score <= 5:
            return 3  # 中等任务
        elif score <= 7:
            return 2  # 较复杂任务
        else:
            return 1  # 非常复杂任务用最强模型

    def is_available(self) -> bool:
        """
        is_available 检查本地 AI 服务是否可用。

        Returns:
            是否可用
        """
        try:
            response = self._client.get(f"{self._base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def __del__(self):
        """__del__ 清理资源。"""
        if hasattr(self, "_client"):
            self._client.close()

