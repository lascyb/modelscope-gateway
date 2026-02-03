"""
UsageTracker 负责追踪和持久化每个模型的 API 调用次数。

使用场景：
- 记录每日每个模型的使用量
- 检查是否超出配额限制
- 持久化到 JSON 文件 (usage/YYYY-MM-DD.json)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional


class UsageTracker:
    """UsageTracker 追踪每个模型的 API 调用次数并持久化到 JSON 文件。"""

    def __init__(self, usage_dir: str = "usage"):
        """
        UsageTracker 初始化追踪器。

        Args:
            usage_dir: 使用量记录文件存储目录
        """
        self._usage_dir = Path(usage_dir)
        self._usage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._cache: Dict[str, Dict] = {}
        self._current_date: Optional[str] = None

    def _get_today_str(self) -> str:
        """_get_today_str 获取今天的日期字符串 (YYYY-MM-DD)。"""
        return datetime.now().strftime("%Y-%m-%d")

    def _get_usage_file_path(self, date_str: str) -> Path:
        """_get_usage_file_path 获取指定日期的使用量文件路径。"""
        return self._usage_dir / f"{date_str}.json"

    def _load_usage(self, date_str: str) -> Dict[str, int]:
        """
        _load_usage 从文件加载指定日期的使用量数据。

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)

        Returns:
            模型使用量字典 {model_id: count}
        """
        file_path = self._get_usage_file_path(date_str)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_usage(self, date_str: str, usage: Dict[str, int]) -> None:
        """
        _save_usage 保存使用量数据到文件。

        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            usage: 模型使用量字典 {model_id: count}
        """
        file_path = self._get_usage_file_path(date_str)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(usage, f, ensure_ascii=False, indent=2)

    def get_usage(self, model_id: str, date_str: Optional[str] = None) -> int:
        """
        get_usage 获取指定模型的使用次数。

        Args:
            model_id: 模型 ID
            date_str: 日期字符串，默认为今天

        Returns:
            使用次数
        """
        date_str = date_str or self._get_today_str()

        with self._lock:
            if self._current_date != date_str:
                self._cache = self._load_usage(date_str)
                self._current_date = date_str

            return self._cache.get(model_id, 0)

    def get_all_usage(self, date_str: Optional[str] = None) -> Dict[str, int]:
        """
        get_all_usage 获取指定日期所有模型的使用量。

        Args:
            date_str: 日期字符串，默认为今天

        Returns:
            所有模型使用量字典 {model_id: count}
        """
        date_str = date_str or self._get_today_str()

        with self._lock:
            if self._current_date != date_str:
                self._cache = self._load_usage(date_str)
                self._current_date = date_str

            return dict(self._cache)

    def get_total_usage(self, date_str: Optional[str] = None) -> int:
        """
        get_total_usage 获取指定日期的总使用量。

        Args:
            date_str: 日期字符串，默认为今天

        Returns:
            总使用次数
        """
        return sum(self.get_all_usage(date_str).values())

    def increment(self, model_id: str, count: int = 1) -> int:
        """
        increment 增加指定模型的使用次数。

        Args:
            model_id: 模型 ID
            count: 增加的次数，默认为 1

        Returns:
            更新后的使用次数
        """
        date_str = self._get_today_str()

        with self._lock:
            if self._current_date != date_str:
                self._cache = self._load_usage(date_str)
                self._current_date = date_str

            current = self._cache.get(model_id, 0)
            self._cache[model_id] = current + count
            self._save_usage(date_str, self._cache)

            return self._cache[model_id]

    def can_use(self, model_id: str, model_limit: int) -> bool:
        """
        can_use 检查指定模型是否还有可用配额。

        Args:
            model_id: 模型 ID
            model_limit: 模型的每日限制

        Returns:
            是否可以使用
        """
        return self.get_usage(model_id) < model_limit

    def get_remaining(self, model_id: str, model_limit: int) -> int:
        """
        get_remaining 获取指定模型的剩余配额。

        Args:
            model_id: 模型 ID
            model_limit: 模型的每日限制

        Returns:
            剩余可用次数
        """
        return max(0, model_limit - self.get_usage(model_id))

