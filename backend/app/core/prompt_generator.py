from app.schemas.agent import AgentAction, LightMode


def build_strategy_message(
    current_light_mode: LightMode,
    action: AgentAction,
    trigger_popup: bool,
    in_cooldown: bool = False,
) -> str:
    if in_cooldown:
        return "系统仍处于冷却时间内，本次不重复触发调光弹窗。"

    if trigger_popup and action == AgentAction.switch_to_adjustment_light:
        return "检测到近一段时间情绪状态偏低，是否切换至光辅助调节模式？"

    if trigger_popup and action == AgentAction.switch_to_normal_office_light:
        return "检测到当前情绪状态已趋于稳定，是否调回正常办公光？"

    if current_light_mode == LightMode.adjustment_light:
        return "当前仍建议维持光辅助调节模式。"

    return "当前情绪状态较稳定，建议维持正常办公光。"

