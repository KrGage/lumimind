"""
提示词注册初始化
================
将所有提示词注册到全局 PromptRegistry，支持版本管理和 A/B 测试。

用法:
    from prompts import register_all_prompts
    register_all_prompts()
"""

from agent_framework.prompt_registry import get_prompt_registry

from .prompt_texts import (
    LUMIMIND_SYSTEM_PROMPT,
    LIGHTING_DESIGN_PROMPT,
    EMOTION_ANALYSIS_PROMPT,
    HEALTH_ADVICE_PROMPT,
)


def register_all_prompts() -> None:
    """将所有提示词注册到全局注册中心。"""

    registry = get_prompt_registry()

    # ── LumiMind 系统提示词 ──
    registry.register(
        "lumimind_system",
        version="1.0",
        template=LUMIMIND_SYSTEM_PROMPT,
        description="LumiMind 医学健康智能助手默认系统提示词",
    )

    # ── 光环境设计提示词 ──
    registry.register(
        "lighting_design",
        version="1.0",
        template=LIGHTING_DESIGN_PROMPT,
        description="光环境设计领域提示词",
    )

    # ── 情绪分析提示词 ──
    registry.register(
        "emotion_analysis",
        version="1.0",
        template=EMOTION_ANALYSIS_PROMPT,
        description="用户情绪状态分析提示词",
    )

    # ── 健康建议提示词 ──
    registry.register(
        "health_advice",
        version="1.0",
        template=HEALTH_ADVICE_PROMPT,
        description="基于健康数据的个性化建议提示词",
    )

    summary = registry.summary()
    print(f"[prompts] 已注册 {summary['total_prompts']} 个提示词，"
          f"共 {summary['total_versions']} 个版本。")
