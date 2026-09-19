"""
Agent Framework — 自定义提示词示例
===================================

此文件演示如何注册和管理自定义提示词。

用法:
    from custom_prompts.example_prompts import register_example_prompts
    register_example_prompts()

    from agent_framework.prompt_registry import get_prompt
    prompt = get_prompt("example_agent").format(user_name="张三", role="助手")
"""

from agent_framework.prompt_registry import get_prompt_registry


def register_example_prompts():
    """注册示例提示词到全局注册中心。"""

    registry = get_prompt_registry()

    # ── 通用对话提示词 ──────────────────────────────────────────

    registry.register(
        "example_agent",
        version="1.0",
        template=(
            "你是一个{role}。\n\n"
            "用户信息: {user_name}\n\n"
            "请用友好、专业的方式回答用户的问题。\n"
            "如果你不确定答案，请诚实说明。\n"
        ),
        description="通用 Agent 对话提示词模板",
    )

    # ── RAG 问答提示词 ──────────────────────────────────────────

    registry.register(
        "example_rag_qa",
        version="1.0",
        template=(
            "你是一个知识问答助手。\n"
            "请基于以下提供的上下文信息回答问题。\n"
            "如果上下文不足以回答问题，请如实说明。\n\n"
            "--- 知识库上下文 ---\n{context}\n--- 上下文结束 ---\n\n"
            "用户问题: {query}\n\n"
            "回答:"
        ),
        description="基于知识库的问答提示词",
    )

    # ── 分析任务提示词 ──────────────────────────────────────────

    registry.register(
        "example_analysis",
        version="1.0",
        template=(
            "你是一个数据分析助手。\n\n"
            "待分析数据:\n{data}\n\n"
            "分析任务: {task}\n\n"
            "请提供:\n"
            "1. 数据概览\n"
            "2. 关键发现\n"
            "3. 建议下一步操作\n\n"
            "分析结果:"
        ),
        description="数据分析任务提示词",
    )

    # ── 多轮对话提示词 ──────────────────────────────────────────

    registry.register(
        "example_multi_turn",
        version="1.0",
        template=(
            "你是一个持续对话助手。\n\n"
            "历史对话:\n{conversation_history}\n\n"
            "当前用户消息: {current_message}\n\n"
            "请结合历史对话上下文，给出连贯的回复。\n"
        ),
        description="多轮对话记忆提示词",
    )

    print(f"已注册 {registry.summary()['total_prompts']} 个示例提示词。")


# ── 按领域组织的提示词注册示例 ──────────────────────────────────

def register_domain_prompts(domain: str):
    """
    按领域注册提示词（演示按需注册模式）。

    Args:
        domain: 领域名称，如 "lighting", "healthcare", "finance"
    """
    registry = get_prompt_registry()

    if domain == "lighting":
        registry.register(
            "lighting_design",
            version="1.0",
            template=(
                "你是一个光环境设计专家。\n\n"
                "空间信息: {space_info}\n"
                "用户需求: {requirements}\n"
                "情绪状态: {emotion_state}\n\n"
                "请提供光环境设计建议，包括:\n"
                "1. 色温建议 (K)\n"
                "2. 照度建议 (lux)\n"
                "3. 动态变化方案\n"
                "4. 情绪调节效果预期\n"
            ),
            description="光环境设计领域提示词",
        )
