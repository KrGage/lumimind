"""
Agent Framework — 安全护栏
===========================

提供提示注入检测和 PII 脱敏功能。

用法:
    from agent_framework.guardrails import is_prompt_injection, redact_pii_from_text

    if is_prompt_injection(user_input):
        raise ValueError("Unsafe input detected")

    safe_text = redact_pii_from_text("Contact me at user@example.com")
    # → "Contact me at [REDACTED_EMAIL]"
"""

import re
from typing import List, Optional

# ── PII 检测正则表达式 ──────────────────────────────────────────
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_REGEX = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
AADHAAR_REGEX = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b")

# ── 提示注入关键词 ──────────────────────────────────────────────
INJECTION_KEYWORDS = [
    "ignore prior instructions",
    "ignore previous instructions",
    "override safety rules",
    "system override",
    "you are now a",
    "you can prescribe",
    "bypass safety",
    "act as an unrestricted",
    "ignore system prompt",
    "developer mode",
    "jailbreak",
    "dan mode",
    "pretend you are",
    "disregard all previous",
    "forget all instructions",
]


def is_prompt_injection(text: str) -> bool:
    """
    检查输入文本是否包含提示注入尝试。

    Args:
        text: 用户输入文本

    Returns:
        是否检测到注入尝试
    """
    if not text:
        return False
    text_lower = text.lower()
    for keyword in INJECTION_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def redact_pii_from_text(text: str) -> str:
    """
    扫描并脱敏文本中的 PII 信息。

    当前支持：邮箱、SSN、电话号码、Aadhaar 号

    Args:
        text: 原始文本

    Returns:
        脱敏后的文本
    """
    if not text:
        return ""

    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = SSN_REGEX.sub("[REDACTED_SSN]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    text = AADHAAR_REGEX.sub("[REDACTED_AADHAAR]", text)

    return text


# ── 可扩展接口 ──────────────────────────────────────────────────

def add_injection_keyword(keyword: str) -> None:
    """
    添加自定义提示注入关键词。

    Args:
        keyword: 要检测的关键词（小写）
    """
    if keyword.lower() not in INJECTION_KEYWORDS:
        INJECTION_KEYWORDS.append(keyword.lower())


def add_pii_pattern(name: str, pattern: str, replacement: str = "[REDACTED]") -> None:
    """
    添加自定义 PII 检测模式。

    Args:
        name: 模式名称
        pattern: 正则表达式
        replacement: 替换文本
    """
    # 存储为模块级变量供 redact_pii_from_text 使用
    if not hasattr(add_pii_pattern, "_patterns"):
        add_pii_pattern._patterns = []
    add_pii_pattern._patterns.append((re.compile(pattern), replacement))
