"""
Agent Framework — 通用提示词注册中心
=====================================

支持版本控制、激活/停用和 A/B 测试的提示词模板管理中心。

用法:
    from agent_framework.prompt_registry import get_prompt_registry, register_prompt, get_prompt

    # 注册提示词
    registry = get_prompt_registry()
    registry.register(
        "my_agent",
        version="1.0",
        template="你是一个{role}助手。用户: {user_name}",
        description="自定义 Agent 系统提示词",
    )

    # 获取并使用
    prompt = get_prompt("my_agent")
    formatted = prompt.format(role="光环境设计", user_name="张三")

从 AI-Healthcare-System 抽取而来，已移除医疗领域默认提示词。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """提示词的单个版本。"""
    name: str
    version: str
    template: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


class PromptRegistry:
    """
    提示词模板管理中心。

    功能：
      - 版本管理（同一提示词可注册多个版本）
      - 激活/停用（随时切换活跃版本）
      - A/B 测试（通过版本号指定）
      - 元数据标注（描述、自定义标签）
    """

    def __init__(self):
        self._prompts: Dict[str, List[PromptVersion]] = {}
        self._active: Dict[str, str] = {}  # name → active version

    # ── 注册 ──────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        version: str,
        template: str,
        description: str = "",
        activate: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptVersion:
        """
        注册一个新提示词版本。

        Args:
            name: 提示词名称（唯一标识）
            version: 版本号（如 "1.0", "1.1"）
            template: 提示词模板（支持 .format() 占位符）
            description: 描述
            activate: 是否自动激活此版本
            metadata: 自定义元数据

        Returns:
            注册的 PromptVersion 实例
        """
        prompt = PromptVersion(
            name=name,
            version=version,
            template=template,
            description=description,
            metadata=metadata or {},
            active=activate,
        )

        if name not in self._prompts:
            self._prompts[name] = []

        # 检查重复版本 — 就地更新
        for existing in self._prompts[name]:
            if existing.version == version:
                existing.template = template
                existing.description = description
                existing.metadata = metadata or {}
                logger.info("Updated prompt: %s v%s", name, version)
                if activate:
                    self._active[name] = version
                return existing

        self._prompts[name].append(prompt)

        if activate:
            self._active[name] = version

        logger.info("Registered prompt: %s v%s (active=%s)", name, version, activate)
        return prompt

    # ── 获取 ──────────────────────────────────────────────────────

    def get(self, name: str, version: Optional[str] = None) -> str:
        """
        获取提示词模板。

        Args:
            name: 提示词名称
            version: 指定版本号，None 则返回当前激活版本

        Returns:
            提示词模板字符串

        Raises:
            KeyError: 提示词或版本不存在
        """
        if name not in self._prompts:
            raise KeyError(f"Unknown prompt: {name}")

        target_version = version or self._active.get(name)
        if not target_version:
            raise KeyError(f"No active version for prompt: {name}")

        for prompt in self._prompts[name]:
            if prompt.version == target_version:
                return prompt.template

        raise KeyError(f"Version {target_version} not found for prompt: {name}")

    def get_info(self, name: str) -> dict:
        """
        获取提示词的元数据及所有版本信息。

        Args:
            name: 提示词名称

        Returns:
            包含 active_version 和 versions 列表的 dict
        """
        if name not in self._prompts:
            raise KeyError(f"Unknown prompt: {name}")

        versions = self._prompts[name]
        active_version = self._active.get(name, "")

        return {
            "name": name,
            "active_version": active_version,
            "versions": [
                {
                    "version": v.version,
                    "description": v.description,
                    "active": v.version == active_version,
                    "created_at": v.created_at,
                    "template_length": len(v.template),
                }
                for v in versions
            ],
        }

    # ── 管理 ──────────────────────────────────────────────────────

    def activate(self, name: str, version: str) -> None:
        """
        激活指定的提示词版本。

        Args:
            name: 提示词名称
            version: 要激活的版本号
        """
        if name not in self._prompts:
            raise KeyError(f"Unknown prompt: {name}")
        found = any(v.version == version for v in self._prompts[name])
        if not found:
            raise KeyError(f"Version {version} not found for prompt: {name}")
        self._active[name] = version
        logger.info("Activated prompt: %s v%s", name, version)

    def deactivate(self, name: str) -> None:
        """
        停用提示词（移除活跃版本标记）。

        Args:
            name: 提示词名称
        """
        self._active.pop(name, None)
        logger.info("Deactivated prompt: %s", name)

    def list_all(self) -> List[dict]:
        """列出所有已注册的提示词及其活跃版本。"""
        return [
            {
                "name": name,
                "active_version": self._active.get(name, ""),
                "total_versions": len(versions),
            }
            for name, versions in self._prompts.items()
        ]

    def summary(self) -> dict:
        """返回注册中心摘要信息。"""
        return {
            "total_prompts": len(self._prompts),
            "total_versions": sum(len(v) for v in self._prompts.values()),
            "prompts": self.list_all(),
        }

    def remove(self, name: str) -> None:
        """
        移除整个提示词及其所有版本。

        Args:
            name: 提示词名称
        """
        self._prompts.pop(name, None)
        self._active.pop(name, None)
        logger.info("Removed prompt: %s", name)


# ── 全局单例 ─────────────────────────────────────────────────────

_registry: Optional[PromptRegistry] = None


def get_prompt_registry() -> PromptRegistry:
    """获取全局 PromptRegistry 单例。"""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry


def get_prompt(name: str, version: Optional[str] = None) -> str:
    """快捷获取提示词模板（从全局注册中心）。"""
    return get_prompt_registry().get(name, version)


def register_prompt(
    name: str,
    version: str,
    template: str,
    description: str = "",
    activate: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> PromptVersion:
    """快捷注册提示词（到全局注册中心）。"""
    return get_prompt_registry().register(
        name, version, template, description, activate, metadata
    )
