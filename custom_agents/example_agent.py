"""
Agent Framework — 自定义 Agent 示例
===================================

此文件演示如何基于 agent_framework 扩展自定义 Agent。
复制此文件并根据需求修改。

示例包含:
  1. CustomSupervisorNode — 自定义路由节点
  2. CustomAnalysisNode — 自定义分析节点
  3. 完整自定义工作流
"""

from typing import Any, Dict, List

from agent_framework.workflow import (
    AgentState,
    AgentWorkflow,
    BaseNode,
    BaseRouter,
    GenerationNode,
    GuardrailNode,
    ResearchNode,
    build_workflow,
)
from agent_framework.llm_wrapper import CoreAIWrapper


# ── 示例 1: 自定义 Supervisor ────────────────────────────────────

class CustomSupervisor(BaseNode):
    """
    自定义主管路由节点。

    根据用户输入决定下一步：搜索、分析、生成，或护栏拒绝。
    """

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        last_msg = messages[-1].content.lower() if messages else ""

        # 自定义护栏关键词
        forbidden = ["violence", "暴力", "illegal"]
        if any(x in last_msg for x in forbidden):
            return {"next_step": "off_topic"}

        # 自定义搜索触发
        if any(w in last_msg for w in ["search", "搜索", "查找"]):
            return {"next_step": "research"}

        # 自定义分析触发
        if any(w in last_msg for w in ["analyze", "分析"]):
            return {"next_step": "analyze"}

        return {"next_step": "respond"}


# ── 示例 2: 自定义分析节点 ───────────────────────────────────────

class CustomAnalysisNode(BaseNode):
    """
    自定义分析节点示例。

    可以在这里放置你的领域逻辑，例如:
      - 调用机器学习模型
      - 数据处理流水线
      - 外部 API 调用
      - 数据库查询
    """

    def __init__(self, model_path: str = ""):
        self.model_path = model_path

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""

        # ----- 在这里添加你的分析逻辑 -----
        # 示例: 模拟分析
        analysis = f"[分析结果] 针对查询 '{query[:50]}...' 的分析完成。"
        # --------------------------------

        return {"analysis_results": analysis}


# ── 示例 3: 自定义路由 ───────────────────────────────────────────

class CustomRouter(BaseRouter):
    """自定义路由：根据消息长度决定走向。"""

    def route(self, state: AgentState) -> str:
        next_step = state.get("next_step", "respond")
        return next_step


# ── 示例 4: 完整自定义工作流 ─────────────────────────────────────

def create_custom_agent(llm=None):
    """
    创建自定义 Agent 工作流示例。

    演示如何:
      1. 使用自定义 Supervisor + 自定义分析节点
      2. 注册自定义提示词
      3. 配置自定义护栏关键词

    Returns:
        编译好的 LangGraph 工作流
    """
    if llm is None:
        llm = CoreAIWrapper()

    # 自定义提示词
    system_prompt = (
        "你是一个专业的 AI 助手。\n"
        "请用中文回答用户的问题。\n"
        "如果包含分析上下文，请基于分析结果回答。\n"
    )

    # 构建工作流
    wf = AgentWorkflow(
        llm=llm,
        system_prompt=system_prompt,
        enable_search=True,
        enable_analysis=True,
        enable_guardrail=True,
        supervisor_kwargs={
            "research_keywords": ["search", "搜索", "查找"],
            "analyze_keywords": ["analyze", "分析"],
            "guard_keywords": ["violence", "暴力"],
        },
        analysis_node=CustomAnalysisNode(),
        generation_kwargs={
            "context_template": (
                "额外上下文：\n"
                "--- 搜索结果 ---\n{search_results}\n"
                "--- 分析结果 ---\n{analysis_results}\n"
            ),
        },
    )

    return wf


# ── 示例 5: 纯手写工作流（最大灵活性） ───────────────────────────

def create_manual_workflow(llm=None):
    """
    纯手写 LangGraph 工作流示例。

    当你需要完全自定义图结构时使用此方式。
    """
    if llm is None:
        llm = CoreAIWrapper()

    nodes = {
        "supervisor": CustomSupervisor(),
        "generate": GenerationNode(llm=llm, system_prompt="你是有帮助的AI助手。"),
        "researcher": ResearchNode(),
        "analyst": CustomAnalysisNode(),
        "guardrail": GuardrailNode(rejection_message="抱歉，我不能处理这个话题。"),
    }

    edge_map = {
        "research": "researcher",
        "analyze": "analyst",
        "respond": "generate",
        "off_topic": "guardrail",
    }

    router = CustomRouter()

    return build_workflow(
        nodes=nodes,
        router=router,
        entry="supervisor",
        edge_map=edge_map,
        terminal_nodes=["guardrail"],
    )
