"""
Agent Framework — LangGraph 通用工作流引擎
============================================

基于 LangGraph StateGraph 的可复用工作流引擎。提供抽象基类用于自定义 Agent 节点和路由逻辑。

核心设计：
  - BaseNode: 工作流节点抽象基类，子类实现 run(state) → dict
  - BaseRouter: 路由抽象基类，子类实现 route(state) → str
  - AgentWorkflow: 通用可配置工作流，接受节点和路由配置构建图

用法:
    # 方式一：使用预置通用工作流
    from agent_framework.workflow import AgentWorkflow
    workflow = AgentWorkflow(llm=wrapper, system_prompt="你是一个助手")
    result = workflow.invoke({"messages": [HumanMessage(content="你好")]})

    # 方式二：自定义节点 + 路由
    from agent_framework.workflow import BaseNode, BaseRouter, build_workflow
    class MyAnalysisNode(BaseNode):
        def run(self, state):
            result = do_something(state)
            return {"analysis_results": result}

    class MyRouter(BaseRouter):
        def route(self, state):
            last_msg = state["messages"][-1].content.lower()
            if "search" in last_msg:
                return "researcher"
            return "generate"

    workflow = build_workflow(
        nodes={"generate": generate_node, "researcher": researcher_node},
        router=MyRouter(),
        entry="supervisor",
    )

从 AI-Healthcare-System 抽取而来，已移除医疗领域特化逻辑。
"""

import logging
import operator
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# ── 通用 Agent 状态定义 ───────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    通用 Agent 工作流状态。

    必需字段（框架自动维护）:
      - messages: 对话消息列表（operator.add 累加）
      - next_step: 路由决策结果

    可选扩展字段（自定义节点按需使用）:
      - search_results: 搜索结果文本
      - analysis_results: 分析结果文本
      - rag_context: 向量知识库检索上下文
      - system_prompt: 自定义系统提示词
    """
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str
    search_results: str
    analysis_results: str
    rag_context: str
    system_prompt: str
    user_context: Dict[str, Any]


def _empty_str(value: object) -> str:
    """安全转换为字符串，None 返回空字符串。"""
    if value is None:
        return ""
    return str(value).strip()


def _compact_str(value: object, limit: int = 700) -> str:
    """压缩文本到限制长度。"""
    text = " ".join(_empty_str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


# ── 抽象基类：节点 ────────────────────────────────────────────────

class BaseNode(ABC):
    """
    工作流节点抽象基类。

    子类必须实现 run(state) 方法，返回一个 dict 用于更新 AgentState。

    用法:
        class MyGreetingNode(BaseNode):
            def run(self, state: AgentState) -> dict:
                name = state.get("user_context", {}).get("name", "User")
                return {"messages": [AIMessage(content=f"Hello {name}!")]}
    """

    @abstractmethod
    def run(self, state: AgentState) -> dict:
        """
        执行节点逻辑。

        Args:
            state: 当前工作流状态

        Returns:
            包含要更新的状态字段的 dict（如 {"messages": [...], "next_step": "..."}）
        """
        ...

    def __call__(self, state: AgentState) -> dict:
        """使节点可被 LangGraph 直接调用。"""
        return self.run(state)


# ── 抽象基类：路由 ────────────────────────────────────────────────

class BaseRouter(ABC):
    """
    工作流路由抽象基类。

    子类必须实现 route(state) 方法，返回一个字符串键，用于条件边映射。

    用法:
        class MyRouter(BaseRouter):
            def route(self, state: AgentState) -> str:
                last = state["messages"][-1].content.lower()
                if "search" in last: return "research"
                return "generate"
    """

    @abstractmethod
    def route(self, state: AgentState) -> str:
        """
        决定下一步路由。

        Args:
            state: 当前工作流状态

        Returns:
            路由键（字符串），对应条件边中映射的节点名
        """
        ...

    def __call__(self, state: AgentState) -> str:
        return self.route(state)


# ── 预置节点实现 ──────────────────────────────────────────────────

class SupervisorNode(BaseNode):
    """
    默认主管节点：基于关键词的轻量路由（零 LLM 调用开销）。

    可自定义：
      - research_keywords: 触发搜索的关键词列表
      - analyze_keywords: 触发分析的关键词列表
      - guard_keywords: 触发护栏的关键词列表（为空则跳过护栏）
    """

    def __init__(
        self,
        research_keywords: Optional[List[str]] = None,
        analyze_keywords: Optional[List[str]] = None,
        guard_keywords: Optional[List[str]] = None,
    ):
        self.research_keywords = research_keywords or [
            "search", "搜索", "查找", "检索", "latest", "news", "最新",
            "treatment", "research", "study", "recent",
        ]
        self.analyze_keywords = analyze_keywords or [
            "analyze", "分析", "predict", "risk", "评估",
        ]
        self.guard_keywords = guard_keywords or []

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        last_msg = messages[-1].content.lower() if messages else ""

        # 护栏检查
        if self.guard_keywords:
            if any(kw in last_msg for kw in self.guard_keywords):
                return {"next_step": "guardrail"}

        # 搜索路由
        if any(w in last_msg for w in self.research_keywords):
            return {"next_step": "research"}

        # 分析路由
        if any(w in last_msg for w in self.analyze_keywords):
            return {"next_step": "analyze"}

        # 默认：直接生成
        return {"next_step": "respond"}


class GenerationNode(BaseNode):
    """
    默认生成节点：使用 LLM 包装器生成最终响应。

    组装系统提示词、对话消息和上下文信息，然后调用 LLM 生成回复。
    """

    def __init__(
        self,
        llm: Any,  # CoreAIWrapper 或任何 .invoke(messages) → AIMessage 的对象
        system_prompt: str = "You are a helpful AI assistant.",
        context_template: Optional[str] = None,
    ):
        """
        Args:
            llm: LangChain 兼容的 LLM 包装器（有 .invoke() 方法）
            system_prompt: 默认系统提示词（会被 state 中的 system_prompt 覆盖）
            context_template: 上下文注入模板，支持 {search_results}, {analysis_results}, {rag_context}
        """
        self.llm = llm
        self.default_system_prompt = system_prompt
        self.context_template = context_template or (
            "Additional context:\n"
            "--- SEARCH RESULTS ---\n{search_results}\n\n"
            "--- ANALYSIS RESULTS ---\n{analysis_results}\n\n"
            "--- RAG CONTEXT ---\n{rag_context}\n\n"
        )

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        system_prompt = state.get("system_prompt", self.default_system_prompt)

        # 构建上下文注入
        search = _empty_str(state.get("search_results"))
        analysis = _empty_str(state.get("analysis_results"))
        rag = _empty_str(state.get("rag_context"))

        context_block = ""
        if search or analysis or rag:
            context_block = self.context_template.format(
                search_results=search or "N/A",
                analysis_results=analysis or "N/A",
                rag_context=rag or "N/A",
            )
            system_prompt += "\n\n" + context_block

        final_messages = [SystemMessage(content=system_prompt)] + list(messages)
        response = self.llm.invoke(final_messages)
        return {"messages": [response]}


class ResearchNode(BaseNode):
    """
    默认搜索节点：调用工具注册中心中的搜索工具。

    需配合 ToolRegistry 使用。搜索结果存入 state["search_results"]。
    """

    def __init__(self, search_func: Optional[Callable] = None):
        """
        Args:
            search_func: 搜索函数，接受 query 字符串，返回结果字符串。
                         为 None 时从 tools.py 的 ToolRegistry 获取。
        """
        self._search_func = search_func

    @property
    def search_func(self) -> Callable:
        if self._search_func is not None:
            return self._search_func
        from .tools import get_tool_registry
        return get_tool_registry().get("web_search")

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""
        logger.info("Researching: %s", query[:100])
        try:
            results = self.search_func(query)
        except Exception as e:
            logger.error("Search failed: %s", e)
            results = "Search temporarily unavailable."
        return {"search_results": results}


class GuardrailNode(BaseNode):
    """
    默认护栏节点：当话题超出范围时给出标准拒绝回复。
    """

    def __init__(self, rejection_message: Optional[str] = None):
        self.rejection_message = rejection_message or (
            "I apologize, but I can only assist with topics within my designated scope. "
            "Please ask me something related to my area of expertise."
        )

    def run(self, state: AgentState) -> dict:
        return {"messages": [AIMessage(content=self.rejection_message)]}


class AnalystNode(BaseNode):
    """
    默认分析节点：由用户提供分析逻辑。

    如果不覆盖 run 方法，则直接透传空分析结果。
    """

    def run(self, state: AgentState) -> dict:
        # 默认不做分析，子类可覆盖
        return {"analysis_results": ""}


# ── 工作流构建器 ──────────────────────────────────────────────────

def build_workflow(
    nodes: Dict[str, BaseNode],
    router: BaseRouter,
    entry: str = "supervisor",
    edge_map: Optional[Dict[str, str]] = None,
    terminal_nodes: Optional[List[str]] = None,
    checkpointer: Optional[Any] = None,
) -> Any:
    """
    构建并编译 LangGraph 工作流。

    Args:
        nodes: 节点名 → BaseNode 实例的映射。
               至少需包含 "supervisor"（或自定义入口名称）节点。
        router: 路由实例，返回路由键。
        entry: 入口节点名称，默认 "supervisor"。
        edge_map: 路由键 → 目标节点名称的映射。
                 例如: {"research": "researcher", "analyze": "analyst", "respond": "generate"}
        terminal_nodes: 直接结束的节点名称列表（如 ["guardrail"]），
                        这些节点执行后直接到 END。
                        其他不在 edge_map 值中的节点也会默认到 END。
        checkpointer: LangGraph checkpointer（如 SqliteSaver），
                      配合 config={"configurable": {"thread_id": session_id}}
                      实现按会话持久化的短期记忆；None 则不持久化。

    Returns:
        编译后的 LangGraph StateGraph 实例（可调用 .invoke(state)）
    """
    if terminal_nodes is None:
        terminal_nodes = ["guardrail"]

    workflow = StateGraph(AgentState)

    # 添加所有节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 设置入口
    workflow.set_entry_point(entry)

    # 设置条件边
    if edge_map:
        workflow.add_conditional_edges(entry, router, edge_map)

    # 设置各节点的出边：
    #   - 终端节点（guardrail 等）→ END
    #   - 生成节点（generate）→ END
    #   - 中间节点（researcher/analyst 等路由目标）→ 生成节点，再由生成节点结束
    #     （若图中不存在 generate 则退化为直接 END）
    all_targets = set(edge_map.values()) if edge_map else set()
    generator = "generate" if "generate" in nodes else None
    for node_name in nodes:
        if node_name == entry or node_name == generator:
            continue
        if node_name in terminal_nodes:
            workflow.add_edge(node_name, END)
        elif node_name in all_targets and generator:
            # 中间节点执行完后进入生成节点
            workflow.add_edge(node_name, generator)
        else:
            workflow.add_edge(node_name, END)

    if generator:
        workflow.add_edge(generator, END)

    return workflow.compile(checkpointer=checkpointer)


# ── 开箱即用的通用 Agent 工作流 ──────────────────────────────────

class AgentWorkflow:
    """
    通用 Agent 工作流，可直接使用或作为基类扩展。

    默认工作流拓扑:
        supervisor → (research → generate) / (analyze → generate) / (generate) / (guardrail → END)

    用法:
        from agent_framework.workflow import AgentWorkflow
        from agent_framework.llm_wrapper import CoreAIWrapper

        llm = CoreAIWrapper()
        wf = AgentWorkflow(
            llm=llm,
            system_prompt="你是一个光环境设计助手。",
        )
        result = wf.invoke({
            "messages": [HumanMessage(content="如何设计适合阅读的光环境？")]
        })
    """

    def __init__(
        self,
        llm: Any,
        system_prompt: str = "You are a helpful AI assistant.",
        enable_search: bool = True,
        enable_analysis: bool = False,
        enable_guardrail: bool = True,
        supervisor_kwargs: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        analysis_node: Optional[BaseNode] = None,
        use_checkpointer: bool = False,
    ):
        """
        Args:
            llm: LangChain 兼容的 LLM 包装器
            system_prompt: 系统提示词
            enable_search: 是否启用网络搜索节点
            enable_analysis: 是否启用分析节点
            enable_guardrail: 是否启用护栏节点
            supervisor_kwargs: 传给 SupervisorNode 的额外参数
            generation_kwargs: 传给 GenerationNode 的额外参数
            analysis_node: 自定义分析节点（覆盖默认空节点）
            use_checkpointer: 是否接入 SqliteSaver checkpointer（data/checkpoints.db），
                              开启后 invoke 可传 config 指定 thread_id 实现多轮记忆
        """
        # 构建节点
        self._nodes: Dict[str, BaseNode] = {}

        # Supervisor
        sup_kw = supervisor_kwargs or {}
        self._nodes["supervisor"] = SupervisorNode(**sup_kw)

        # Generation (always)
        gen_kw = generation_kwargs or {}
        self._nodes["generate"] = GenerationNode(llm=llm, system_prompt=system_prompt, **gen_kw)

        # Guardrail
        if enable_guardrail:
            self._nodes["guardrail"] = GuardrailNode()

        # 构建路由映射
        self._edge_map = {}
        self._terminal = []

        if enable_guardrail:
            self._edge_map["off_topic"] = "guardrail"
            self._terminal.append("guardrail")

        if enable_search:
            self._nodes["researcher"] = ResearchNode()
            self._edge_map["research"] = "researcher"

        if enable_analysis:
            self._nodes["analyst"] = analysis_node or AnalystNode()
            self._edge_map["analyze"] = "analyst"

        self._edge_map["respond"] = "generate"

        # 构建路由：SupervisorNode 是节点（__call__ 返回状态 dict），
        # 条件边需要返回路由字符串，因此用 BaseRouter 适配器包一层
        class _SupervisorRouter(BaseRouter):
            def __init__(self, supervisor: SupervisorNode):
                self._supervisor = supervisor

            def route(self, state: AgentState) -> str:
                return self._supervisor.run(state).get("next_step", "respond")

        self._router = _SupervisorRouter(self._nodes["supervisor"])

        # 编译工作流（可选接入 checkpointer 实现短期记忆持久化）
        self._checkpointer = None
        if use_checkpointer:
            try:
                from .checkpointer import get_checkpointer
                self._checkpointer = get_checkpointer()
            except Exception as e:
                logger.warning("Failed to init checkpointer, workflow will be stateless: %s", e)

        self._compiled = build_workflow(
            nodes=self._nodes,
            router=self._router,
            entry="supervisor",
            edge_map=self._edge_map,
            terminal_nodes=self._terminal,
            checkpointer=self._checkpointer,
        )

    def invoke(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> AgentState:
        """执行工作流。

        Args:
            state: 初始状态（至少需包含 messages 字段）
            config: LangGraph 调用配置；接入 checkpointer 后可传
                    {"configurable": {"thread_id": session_id}} 实现按会话记忆

        Returns:
            更新后的状态（包含 AI 回复消息）
        """
        if config is not None:
            return self._compiled.invoke(state, config=config)
        return self._compiled.invoke(state)

    def add_node(self, name: str, node: BaseNode, after: Optional[str] = None, to_end: bool = True):
        """
        动态添加自定义节点（重建工作流图）。

        Args:
            name: 节点名称
            node: 节点实例
            after: 在哪个 Supervisor 路由键之后执行此节点
            to_end: 此节点执行后是否直接结束
        """
        self._nodes[name] = node
        if after:
            self._edge_map[after] = name
        # 重建工作流
        terminal = self._terminal.copy()
        if to_end and name not in terminal:
            terminal.append(name)
        self._compiled = build_workflow(
            nodes=self._nodes,
            router=self._router,
            entry="supervisor",
            edge_map=self._edge_map,
            terminal_nodes=terminal,
        )

    @property
    def nodes(self) -> Dict[str, BaseNode]:
        return self._nodes

    @property
    def edge_map(self) -> Dict[str, str]:
        return self._edge_map
