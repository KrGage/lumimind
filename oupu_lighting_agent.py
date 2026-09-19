"""
Oupu Lighting Agent — 光环境 + 情绪感知智能体
=============================================

基于 agent_framework 的光环境智能调控 Agent 集成示例。

功能:
  - 情绪感知：分析用户情绪状态（通过 ECG/文本/问卷等信号）
  - 光环境建议：基于情绪状态推荐色温、照度、动态方案
  - 知识库检索：从论文和研究中检索光学-情绪相关知识
  - 可扩展自定义节点和分析逻辑

用法:
    from oupu_lighting_agent import OupuLightingAgent

    agent = OupuLightingAgent()
    result = agent.chat("我今天感到很焦虑，什么样的光线可以帮助我放松？")
    print(result["messages"][-1].content)
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from agent_framework.workflow import (
    AgentState,
    AgentWorkflow,
    BaseNode,
    BaseRouter,
    GenerationNode,
    ResearchNode,
    build_workflow,
)
from agent_framework.llm_wrapper import CoreAIWrapper
from agent_framework.prompt_registry import get_prompt_registry


# ── 光环境系统提示词 ──────────────────────────────────────────────

LIGHTING_SYSTEM_PROMPT = """你是一个智能光环境设计助手，专精于通过光照调节人的情绪和生理状态。

## 你的核心能力
1. 分析用户的情绪状态，推荐合适的光环境方案
2. 解释不同光照参数（色温、照度、光谱）对情绪和生理的影响
3. 基于科学研究提供光照建议

## 光照-情绪映射参考
- 暖白光 (2700K-3000K): 放松、舒适、适合休息和睡眠前
- 中性白光 (3500K-4500K): 专注、清醒、适合工作和阅读
- 冷白光 (5000K-6500K): 提神、警觉、适合早晨和需要高度集中的任务
- 动态光照: 模拟自然光变化，调节昼夜节律

## 照度建议
- 放松场景: 50-150 lux
- 一般活动: 200-500 lux
- 精细工作: 500-1000 lux

## 回答要求
- 使用中文回答
- 提供具体的光环境参数建议（色温、照度、时长）
- 解释建议的科学依据
- 如涉及情绪数据，基于数据给出个性化建议

## 额外上下文
{context_block}
"""


# ── 情绪分析节点 ──────────────────────────────────────────────────

class EmotionAnalysisNode(BaseNode):
    """
    情绪分析节点。

    分析用户输入中的情绪线索（文本情感、关键词匹配），
    也可以扩展为接入 ECG 数据处理等生物信号分析。

    扩展方式:
      1. 接入 ML 模型进行情感分类
      2. 接入 ECG 数据处理模块
      3. 接入问卷评分系统
    """

    # 情绪关键词映射
    EMOTION_KEYWORDS = {
        "焦虑": ["焦虑", "紧张", "不安", "担心", "忧虑", "anxiety", "stress"],
        "抑郁": ["抑郁", "低落", "消沉", "悲伤", "难过", "depression", "sad"],
        "疲劳": ["疲劳", "累", "困", "疲惫", "乏力", "fatigue", "tired"],
        "愤怒": ["愤怒", "生气", "烦躁", "恼怒", "angry", "irritated"],
        "平静": ["平静", "放松", "舒适", "安心", "calm", "relaxed"],
        "专注": ["专注", "集中", "注意", "focus", "concentrate"],
        "兴奋": ["兴奋", "激动", "开心", "高兴", "excited", "happy"],
    }

    def __init__(self, ecg_data_path: Optional[str] = None):
        """
        Args:
            ecg_data_path: ECG 数据文件路径（可选，用于生物信号分析）
        """
        self.ecg_data_path = ecg_data_path

    def analyze_emotion_from_text(self, text: str) -> Dict[str, float]:
        """基于关键词匹配的简单文本情绪分析。"""
        scores = {}
        text_lower = text.lower()
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[emotion] = min(score / len(keywords) * 3, 1.0)
        return scores

    def get_lighting_recommendation(self, emotions: Dict[str, float]) -> str:
        """根据情绪状态生成光照建议。"""
        if not emotions:
            return "未检测到明确情绪信号，建议使用中性白光（4000K, 300 lux）维持舒适状态。"

        dominant = max(emotions, key=emotions.get)
        recommendations = {
            "焦虑": "建议使用暖白光（2700K-3000K），照度 100-200 lux，配合缓慢动态渐变。暖色调有助于降低焦虑水平。",
            "抑郁": "建议使用冷白光（5000K-6500K），照度 500-1000 lux，早晨使用 30 分钟。高色温光照有助于改善情绪。",
            "疲劳": "建议从暖白渐变到冷白（3000K→5000K），照度 300-500 lux，模拟日出效果唤醒身体。",
            "愤怒": "建议使用柔和暖白光（2700K），照度 100-150 lux，静态光照。低强度暖光有助于情绪平复。",
            "平静": "维持当前状态。建议使用中性白光（4000K），照度 200-300 lux。",
            "专注": "建议使用冷白光（5000K-5500K），照度 500-750 lux。高色温促进警觉性和注意力。",
            "兴奋": "如需平静，建议使用暖白光（2700K-3000K），照度 100-200 lux。",
        }
        return recommendations.get(
            dominant,
            f"主导情绪为 {dominant}，建议使用中性白光（4000K, 300 lux）。",
        )

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        query = messages[-1].content if messages else ""

        # 文本情绪分析
        emotions = self.analyze_emotion_from_text(query)

        # 光照建议
        recommendation = self.get_lighting_recommendation(emotions)

        # 组装分析结果
        analysis_parts = []
        if emotions:
            analysis_parts.append(f"检测到的情绪信号: {emotions}")
        analysis_parts.append(f"光照建议: {recommendation}")

        analysis = "\n".join(analysis_parts)

        return {"analysis_results": analysis}


# ── 光环境路由节点 ─────────────────────────────────────────────────

class LightingSupervisor(BaseNode):
    """
    光环境领域路由节点。

    根据用户输入判断:
      - lighting_design: 需要光照方案设计
      - research: 需要搜索最新研究
      - emotion_analyze: 需要情绪分析
      - respond: 直接回答
    """

    def run(self, state: AgentState) -> dict:
        messages = state.get("messages", [])
        last_msg = messages[-1].content.lower() if messages else ""

        # 光环境设计相关
        design_keywords = ["设计", "方案", "推荐", "建议", "参数", "色温", "照度", "design"]
        if any(w in last_msg for w in design_keywords):
            return {"next_step": "lighting_design"}

        # 研究搜索相关
        research_keywords = ["研究", "论文", "最新", "文献", "科学", "research", "study"]
        if any(w in last_msg for w in research_keywords):
            return {"next_step": "research"}

        # 情绪分析相关
        emotion_keywords = ["情绪", "情感", "心情", "感觉", "焦虑", "抑郁", "疲劳", "emotion"]
        if any(w in last_msg for w in emotion_keywords):
            return {"next_step": "emotion_analyze"}

        return {"next_step": "respond"}


# ── 路由适配器 ─────────────────────────────────────────────────────

class LightingRouter(BaseRouter):
    """将 supervisor 的 next_step 映射到对应节点。"""

    def route(self, state: AgentState) -> str:
        return state.get("next_step", "respond")


# ── Oupu Lighting Agent 主类 ────────────────────────────────────────

class OupuLightingAgent:
    """
    光环境 + 情绪感知智能体。

    工作流拓扑:
        lighting_supervisor
          ├── research ──────────→ generate
          ├── emotion_analyze ───→ generate
          ├── lighting_design ───→ generate
          └── respond ───────────→ generate

    用法:
        agent = OupuLightingAgent()
        result = agent.chat("如何设计一个帮助放松的光环境？")
        print(result)
    """

    def __init__(
        self,
        llm=None,
        ecg_data_path: Optional[str] = None,
        enable_search: bool = True,
    ):
        self.llm = llm or CoreAIWrapper()
        self.ecg_data_path = ecg_data_path

        # 注册光环境领域提示词
        self._register_prompts()

        # 构建节点
        self._nodes = {
            "supervisor": LightingSupervisor(),
            "generate": GenerationNode(
                llm=self.llm,
                system_prompt=LIGHTING_SYSTEM_PROMPT,
            ),
            "emotion_analyzer": EmotionAnalysisNode(ecg_data_path=ecg_data_path),
        }

        self._edge_map = {
            "respond": "generate",
            "emotion_analyze": "emotion_analyzer",
            "lighting_design": "generate",
        }

        if enable_search:
            self._nodes["researcher"] = ResearchNode()
            self._edge_map["research"] = "researcher"

        # 编译工作流
        self._compiled = build_workflow(
            nodes=self._nodes,
            router=LightingRouter(),
            entry="supervisor",
            edge_map=self._edge_map,
        )

    def _register_prompts(self):
        """注册光环境领域提示词。"""
        registry = get_prompt_registry()

        registry.register(
            "lighting_emotion",
            version="1.0",
            template=LIGHTING_SYSTEM_PROMPT,
            description="光环境-情绪智能体系统提示词",
        )

        registry.register(
            "lighting_design_task",
            version="1.0",
            template=(
                "你是一个光环境设计师。\n\n"
                "空间类型: {space_type}\n"
                "使用场景: {use_case}\n"
                "用户偏好: {preferences}\n"
                "情绪需求: {emotion_requirements}\n\n"
                "请提供完整的光环境设计方案，包括:\n"
                "1. 主照明参数（色温、照度、显色指数）\n"
                "2. 辅助照明建议\n"
                "3. 动态光照方案（如有）\n"
                "4. 预期效果说明\n"
            ),
            description="光环境设计任务提示词",
        )

    def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        发送消息给 Agent。

        Args:
            message: 用户消息
            system_prompt: 可选自定义系统提示词

        Returns:
            AI 回复文本
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
        }
        if system_prompt:
            initial_state["system_prompt"] = system_prompt

        result = self._compiled.invoke(initial_state)
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        return "AI 暂时无法生成回复。"

    def invoke(self, state: AgentState) -> AgentState:
        """直接调用底层工作流（获取完整状态）。"""
        return self._compiled.invoke(state)

    @property
    def compiled_workflow(self):
        """获取编译后的 LangGraph 工作流。"""
        return self._compiled


# ── 快速启动 ──────────────────────────────────────────────────────


def create_oupu_agent(
    enable_search: bool = True,
    ecg_data_path: Optional[str] = None,
) -> OupuLightingAgent:
    """
    工厂函数：快速创建 Oupu Lighting Agent。

    Args:
        enable_search: 是否启用 Tavily 搜索
        ecg_data_path: ECG 数据路径（可选）

    Returns:
        已初始化的 OupuLightingAgent 实例
    """
    return OupuLightingAgent(
        enable_search=enable_search,
        ecg_data_path=ecg_data_path,
    )


# ── 示例：直接运行 ───────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_oupu_agent()

        # 示例对话
        test_messages = [
            "我今天感到很焦虑，什么样的光线可以帮助我放松？",
            "有没有最新的研究支持光照对情绪的调节作用？",
            "请帮我设计一个适合阅读和工作的书房光环境。",
        ]

        for msg in test_messages:
            print(f"\n{'='*60}")
            print(f"用户: {msg}")
            print(f"{'='*60}")
            response = agent.chat(msg)
            print(f"AI: {response}\n")

    asyncio.run(main())
