"""
Agent Framework — 工具注册中心
===============================

可扩展的工具注册与调用系统。内置 Tavily 网络搜索工具。

用法:
    from agent_framework.tools import ToolRegistry, get_tool_registry, tavily_search

    # 注册自定义工具
    registry = get_tool_registry()
    registry.register("weather", lambda city: f"Weather in {city}: Sunny")

    # 使用 Tavily 搜索
    results = tavily_search("latest AI research")
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


# ── Tavily 搜索工具 ───────────────────────────────────────────────

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

SEARCH_FAILURE_MESSAGE = "Search temporarily unavailable."
SEARCH_UPSTREAM_ERROR_MESSAGE = "Search service returned an error."
SEARCH_KEY_MISSING_MESSAGE = "Tavily API Key not configured. Set TAVILY_API_KEY environment variable."


def tavily_search(
    query: str,
    search_depth: str = "advanced",
    topic: str = "general",
    max_results: int = 3,
    include_answer: bool = True,
) -> str:
    """
    使用 Tavily API 进行实时网络搜索。

    Args:
        query: 搜索查询字符串
        search_depth: 搜索深度 "basic" 或 "advanced"
        topic: 搜索分类 "general" 或 "news"
        max_results: 最大返回结果数
        include_answer: 是否包含 AI 摘要答案

    Returns:
        格式化的搜索结果字符串（包含答案和来源链接）
    """
    if not TAVILY_API_KEY:
        return SEARCH_KEY_MISSING_MESSAGE

    try:
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "max_results": max_results,
        }
        headers = {"content-type": "application/json"}
        resp = requests.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=15)

        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "")
            sources = [r.get("url", "") for r in data.get("results", [])]
            parts = []
            if answer:
                parts.append(f"Answer: {answer}")
            if sources:
                parts.append(f"Sources: {sources}")
            return "\n".join(parts)
        else:
            logger.error("Tavily API error: %s", resp.status_code)
            return SEARCH_UPSTREAM_ERROR_MESSAGE

    except requests.exceptions.Timeout:
        return "Search request timed out."
    except Exception as e:
        logger.error("Tavily search failed: %s", e)
        return SEARCH_FAILURE_MESSAGE


# ── 工具注册中心 ──────────────────────────────────────────────────

class ToolRegistry:
    """
    可扩展的工具注册中心。

    功能:
      - register(name, func): 注册工具函数
      - get(name): 获取工具
      - call(name, *args, **kwargs): 调用工具
      - list_all(): 列出所有已注册工具
      - remove(name): 移除工具
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable, override: bool = False) -> None:
        """
        注册工具函数。

        Args:
            name: 工具名称（唯一标识）
            func: 工具函数（callable）
            override: 是否覆盖已存在的同名工具

        Raises:
            ValueError: 工具名已存在且 override=False
        """
        if name in self._tools and not override:
            raise ValueError(
                f"Tool '{name}' already registered. Use override=True to replace."
            )
        self._tools[name] = func
        logger.info("Registered tool: %s", name)

    def get(self, name: str) -> Callable:
        """
        获取已注册的工具函数。

        Args:
            name: 工具名称

        Returns:
            工具 callable

        Raises:
            KeyError: 工具未注册
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def call(self, name: str, *args, **kwargs) -> Any:
        """
        调用已注册的工具。

        Args:
            name: 工具名称
            *args, **kwargs: 传递给工具函数的参数
        """
        func = self.get(name)
        return func(*args, **kwargs)

    def list_all(self) -> List[str]:
        """列出所有已注册的工具名称。"""
        return list(self._tools.keys())

    def remove(self, name: str) -> None:
        """
        移除工具。

        Args:
            name: 工具名称
        """
        self._tools.pop(name, None)
        logger.info("Removed tool: %s", name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ── 全局单例 ─────────────────────────────────────────────────────

_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """获取全局 ToolRegistry 单例（自动注册 Tavily 搜索）。"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.register("web_search", tavily_search)
    return _registry


def register_tool(name: str, func: Callable, override: bool = False) -> None:
    """快捷注册工具到全局注册中心。"""
    get_tool_registry().register(name, func, override)
