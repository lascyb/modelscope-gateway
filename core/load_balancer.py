"""
LoadBalancer 根据配置顺序和剩余配额选择可用模型。

使用场景：
- 按配置文件中的顺序选择可用模型（越靠前优先级越高）
- 支持按模型层级（tier）选择合适的模型（用于智能路由）
- 自动跳过已耗尽配额的模型（基于 API 响应头的限制信息）
- 支持动态更新模型配置
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from .usage_tracker import UsageTracker

if TYPE_CHECKING:
    from .limits_tracker import LimitsTracker


@dataclass
class ModelConfig:
    """ModelConfig 模型配置数据类。"""

    id: str  # 模型 ID
    name: str  # 模型名称
    tier: int  # 模型层级 {1:最强, 2:次强, 3:中等, 4:轻量}
    enabled: bool  # 是否启用 {true:启用, false:禁用}


class LoadBalancer:
    """LoadBalancer 根据配置顺序和配额进行模型负载均衡。"""

    def __init__(
        self,
        config_path: str = "models_config.json",
        usage_tracker: Optional[UsageTracker] = None,
        limits_tracker: Optional["LimitsTracker"] = None,
    ):
        """
        LoadBalancer 初始化负载均衡器。

        Args:
            config_path: 模型配置文件路径
            usage_tracker: 使用量追踪器实例，如果为 None 则自动创建
            limits_tracker: 限制追踪器实例，用于从 API 响应头获取限制信息
        """
        self._config_path = Path(config_path)
        self._tracker = usage_tracker or UsageTracker()
        self._limits_tracker = limits_tracker
        self._models: List[ModelConfig] = []
        self._smart_routing_config: dict = {}
        self._load_config()

    def _load_config(self) -> None:
        """_load_config 从配置文件加载模型配置，保持原有顺序。"""
        if not self._config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self._config_path}")

        with open(self._config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 加载智能路由配置
        self._smart_routing_config = config.get("smart_routing", {})

        # 按配置文件中的顺序加载
        self._models = [
            ModelConfig(
                id=m["id"],
                name=m["name"],
                tier=m.get("tier", 1),  # 默认为最高层级
                enabled=m.get("enabled", True),
            )
            for m in config.get("models", [])
        ]

    def reload_config(self) -> None:
        """reload_config 重新加载模型配置。"""
        self._load_config()

    def get_available_model(
        self,
        target_tier: Optional[int] = None,
        exclude_models: Optional[List[str]] = None,
    ) -> Optional[ModelConfig]:
        """
        get_available_model 获取当前可用的模型。

        Args:
            target_tier: 目标层级，如果指定则优先选择该层级的模型
            exclude_models: 要排除的模型 ID 列表（用于 429 重试时跳过已失败的模型）

        优先级逻辑：
        1. 如果指定了 target_tier，优先选择该层级的可用模型
        2. 如果目标层级无可用模型，向上（更强）或向下（更弱）寻找
        3. 如果没有指定 target_tier，按配置顺序选择
        4. 检查模型和全局剩余配额
        5. 跳过 exclude_models 中的模型

        Returns:
            可用的模型配置，如果没有可用模型返回 None
        """
        exclude_models = exclude_models or []

        # 检查全局限制
        if self._limits_tracker and not self._limits_tracker.can_use_global():
            return None

        if target_tier is not None:
            return self._get_model_by_tier(target_tier, exclude_models)

        # 默认：按配置顺序选择
        for model in self._models:
            if model.id in exclude_models:
                continue
            if self._is_model_available(model):
                return model

        return None

    def _get_model_by_tier(
        self, target_tier: int, exclude_models: Optional[List[str]] = None
    ) -> Optional[ModelConfig]:
        """
        _get_model_by_tier 按层级选择模型。

        优先选择目标层级，如果不可用则向上/向下寻找。
        """
        exclude_models = exclude_models or []

        # 按层级分组
        tiers = {}
        for model in self._models:
            if model.tier not in tiers:
                tiers[model.tier] = []
            tiers[model.tier].append(model)

        # 优先尝试目标层级
        if target_tier in tiers:
            for model in tiers[target_tier]:
                if model.id in exclude_models:
                    continue
                if self._is_model_available(model):
                    return model

        # 目标层级无可用模型，向下（更弱的模型）寻找
        for tier in sorted(tiers.keys()):
            if tier > target_tier:
                for model in tiers[tier]:
                    if model.id in exclude_models:
                        continue
                    if self._is_model_available(model):
                        return model

        # 仍无可用，向上（更强的模型）寻找
        for tier in sorted(tiers.keys(), reverse=True):
            if tier < target_tier:
                for model in tiers[tier]:
                    if model.id in exclude_models:
                        continue
                    if self._is_model_available(model):
                        return model

        return None

    def _is_model_available(self, model: ModelConfig) -> bool:
        """_is_model_available 检查模型是否可用。"""
        if not model.enabled:
            return False

        if self._limits_tracker:
            if not self._limits_tracker.can_use_model(model.id):
                return False

        return True

    def get_all_models(self) -> List[ModelConfig]:
        """
        get_all_models 获取所有已配置的模型列表。

        Returns:
            按配置顺序的模型列表
        """
        return list(self._models)

    def get_enabled_models(self) -> List[ModelConfig]:
        """
        get_enabled_models 获取所有已启用的模型列表。

        Returns:
            按配置顺序的已启用模型列表
        """
        return [m for m in self._models if m.enabled]

    def get_models_by_tier(self, tier: int) -> List[ModelConfig]:
        """
        get_models_by_tier 获取指定层级的模型列表。

        Args:
            tier: 模型层级

        Returns:
            指定层级的模型列表
        """
        return [m for m in self._models if m.tier == tier]

    def get_model_status(self, model_id: str) -> dict:
        """
        get_model_status 获取指定模型的状态信息。

        Args:
            model_id: 模型 ID

        Returns:
            包含使用量和剩余配额的状态字典
        """
        model = None
        priority = 0
        for idx, m in enumerate(self._models):
            if m.id == model_id:
                model = m
                priority = idx + 1
                break

        if not model:
            return {"error": "模型不存在"}

        usage = self._tracker.get_usage(model_id)

        # 从限制追踪器获取限制信息
        daily_limit = None
        remaining = None
        if self._limits_tracker:
            daily_limit = self._limits_tracker.get_model_limit(model_id)
            remaining = self._limits_tracker.get_model_remaining(model_id)

        return {
            "model_id": model_id,
            "name": model.name,
            "tier": model.tier,
            "priority": priority,
            "daily_limit": daily_limit,
            "usage": usage,
            "remaining": remaining,
            "enabled": model.enabled,
            "available": model.enabled and (remaining is None or remaining > 0),
        }

    def get_all_status(self) -> List[dict]:
        """
        get_all_status 获取所有模型的状态信息。

        Returns:
            所有模型的状态列表
        """
        return [self.get_model_status(m.id) for m in self._models]

    def get_global_status(self) -> dict:
        """
        get_global_status 获取全局使用状态。

        Returns:
            全局状态字典
        """
        total_usage = self._tracker.get_total_usage()

        # 从限制追踪器获取全局限制
        global_limit = None
        global_remaining = None
        if self._limits_tracker:
            global_limit = self._limits_tracker.get_global_limit()
            global_remaining = self._limits_tracker.get_global_remaining()

        return {
            "global_daily_limit": global_limit,
            "total_usage": total_usage,
            "remaining": global_remaining,
            "smart_routing_enabled": self._smart_routing_config.get("enabled", False),
            "models": self.get_all_status(),
        }

    def get_smart_routing_config(self) -> dict:
        """get_smart_routing_config 获取智能路由配置。"""
        return dict(self._smart_routing_config)

    def record_usage(self, model_id: str) -> int:
        """
        record_usage 记录一次模型使用。

        Args:
            model_id: 模型 ID

        Returns:
            更新后的使用次数
        """
        return self._tracker.increment(model_id)

    @property
    def tracker(self) -> UsageTracker:
        """tracker 获取使用量追踪器实例。"""
        return self._tracker

    @property
    def limits_tracker(self) -> Optional["LimitsTracker"]:
        """limits_tracker 获取限制追踪器实例。"""
        return self._limits_tracker
