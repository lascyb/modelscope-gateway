"""
LimitsTracker 追踪从 API 响应头获取的配额限制信息。

使用场景：
- 从 API 响应头解析限制信息
- 持久化限制信息到 JSON 文件 (limits/YYYY-MM-DD.json)
- 提供模型和全局的限制查询
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional


class LimitsTracker:
    """LimitsTracker 追踪从 API 响应头获取的配额限制。"""

    def __init__(self, limits_dir: str = "limits"):
        """
        LimitsTracker 初始化限制追踪器。

        Args:
            limits_dir: 限制信息存储目录
        """
        self._limits_dir = Path(limits_dir)
        self._limits_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._cache: Dict[str, Dict] = {}
        self._current_date: Optional[str] = None

    def _get_today_str(self) -> str:
        """_get_today_str 获取今天的日期字符串 (YYYY-MM-DD)。"""
        return datetime.now().strftime("%Y-%m-%d")

    def _get_limits_file_path(self, date_str: str) -> Path:
        """_get_limits_file_path 获取指定日期的限制文件路径。"""
        return self._limits_dir / f"{date_str}.json"

    def _load_limits(self, date_str: str) -> Dict:
        """
        _load_limits 从文件加载指定日期的限制数据。

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)

        Returns:
            限制数据字典
        """
        file_path = self._get_limits_file_path(date_str)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"global": {}, "models": {}}

    def _save_limits(self, date_str: str, limits: Dict) -> None:
        """
        _save_limits 保存限制数据到文件。

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            limits: 限制数据字典
        """
        file_path = self._get_limits_file_path(date_str)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(limits, f, ensure_ascii=False, indent=2)

    def _ensure_cache(self) -> None:
        """_ensure_cache 确保缓存是最新的。"""
        date_str = self._get_today_str()
        if self._current_date != date_str:
            self._cache = self._load_limits(date_str)
            self._current_date = date_str

    def update_from_headers(
        self, model_id: str, headers: Dict[str, str]
    ) -> Dict[str, int]:
        """
        update_from_headers 从 API 响应头更新限制信息。

        ModelScope API 响应头格式：
        - modelscope-ratelimit-requests-limit: 用户当天限额
        - modelscope-ratelimit-requests-remaining: 用户当天剩余额度
        - modelscope-ratelimit-model-requests-limit: 模型当天限额
        - modelscope-ratelimit-model-requests-remaining: 模型当天剩余额度

        Args:
            model_id: 模型 ID
            headers: API 响应头字典

        Returns:
            解析后的限制信息
        """
        date_str = self._get_today_str()

        with self._lock:
            self._ensure_cache()

            # 解析模型级别限制
            model_limit = self._parse_int_header(
                headers, "modelscope-ratelimit-model-requests-limit"
            )
            model_remaining = self._parse_int_header(
                headers, "modelscope-ratelimit-model-requests-remaining"
            )

            # 解析用户全局限制
            global_limit = self._parse_int_header(
                headers, "modelscope-ratelimit-requests-limit"
            )
            global_remaining = self._parse_int_header(
                headers, "modelscope-ratelimit-requests-remaining"
            )

            # 更新模型限制
            if model_limit is not None:
                if "models" not in self._cache:
                    self._cache["models"] = {}
                self._cache["models"][model_id] = {
                    "limit": model_limit,
                    "remaining": model_remaining,
                    "usage": model_limit - model_remaining if model_remaining is not None else None,
                    "updated_at": datetime.now().isoformat(),
                }

            # 更新全局限制
            if global_limit is not None:
                self._cache["global"] = {
                    "limit": global_limit,
                    "remaining": global_remaining,
                    "usage": global_limit - global_remaining if global_remaining is not None else None,
                    "updated_at": datetime.now().isoformat(),
                }

            self._save_limits(date_str, self._cache)

            return {
                "model_limit": model_limit,
                "model_remaining": model_remaining,
                "global_limit": global_limit,
                "global_remaining": global_remaining,
            }

    def _parse_int_header(
        self, headers: Dict[str, str], key: str
    ) -> Optional[int]:
        """_parse_int_header 解析响应头中的整数值。"""
        # 尝试多种大小写格式
        for k in [key, key.lower(), key.upper(), key.title()]:
            if k in headers:
                try:
                    return int(headers[k])
                except (ValueError, TypeError):
                    pass
        return None

    def get_model_limit(self, model_id: str) -> Optional[int]:
        """
        get_model_limit 获取模型的每日限制。

        Args:
            model_id: 模型 ID

        Returns:
            每日限制，如果未知返回 None
        """
        with self._lock:
            self._ensure_cache()
            model_info = self._cache.get("models", {}).get(model_id, {})
            return model_info.get("limit")

    def get_model_remaining(self, model_id: str) -> Optional[int]:
        """
        get_model_remaining 获取模型的剩余配额。

        Args:
            model_id: 模型 ID

        Returns:
            剩余配额，如果未知返回 None
        """
        with self._lock:
            self._ensure_cache()
            model_info = self._cache.get("models", {}).get(model_id, {})
            return model_info.get("remaining")

    def get_global_limit(self) -> Optional[int]:
        """
        get_global_limit 获取全局每日限制。

        Returns:
            全局每日限制，如果未知返回 None
        """
        with self._lock:
            self._ensure_cache()
            return self._cache.get("global", {}).get("limit")

    def get_global_remaining(self) -> Optional[int]:
        """
        get_global_remaining 获取全局剩余配额。

        Returns:
            全局剩余配额，如果未知返回 None
        """
        with self._lock:
            self._ensure_cache()
            return self._cache.get("global", {}).get("remaining")

    def get_all_limits(self) -> Dict:
        """
        get_all_limits 获取所有限制信息。

        Returns:
            包含全局和模型限制的字典
        """
        with self._lock:
            self._ensure_cache()
            return dict(self._cache)

    def can_use_model(self, model_id: str) -> bool:
        """
        can_use_model 检查模型是否还有可用配额。

        Args:
            model_id: 模型 ID

        Returns:
            是否可用（如果未知限制，默认可用）
        """
        remaining = self.get_model_remaining(model_id)
        if remaining is None:
            return True  # 未知限制时默认可用
        return remaining > 0

    def can_use_global(self) -> bool:
        """
        can_use_global 检查全局是否还有可用配额。

        Returns:
            是否可用（如果未知限制，默认可用）
        """
        remaining = self.get_global_remaining()
        if remaining is None:
            return True  # 未知限制时默认可用
        return remaining > 0

    def mark_model_exhausted(self, model_id: str) -> None:
        """
        mark_model_exhausted 标记模型配额已用尽（收到 429 时调用）。

        Args:
            model_id: 模型 ID
        """
        date_str = self._get_today_str()

        with self._lock:
            self._ensure_cache()

            if "models" not in self._cache:
                self._cache["models"] = {}

            existing = self._cache["models"].get(model_id, {})
            self._cache["models"][model_id] = {
                "limit": existing.get("limit"),
                "remaining": 0,  # 标记为 0
                "usage": existing.get("limit"),
                "exhausted": True,
                "exhausted_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            self._save_limits(date_str, self._cache)

    def mark_global_exhausted(self) -> None:
        """mark_global_exhausted 标记全局配额已用尽（收到 429 时调用）。"""
        date_str = self._get_today_str()

        with self._lock:
            self._ensure_cache()

            existing = self._cache.get("global", {})
            self._cache["global"] = {
                "limit": existing.get("limit"),
                "remaining": 0,
                "usage": existing.get("limit"),
                "exhausted": True,
                "exhausted_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            self._save_limits(date_str, self._cache)

