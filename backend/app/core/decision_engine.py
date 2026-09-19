from app.config import settings
from app.schemas.agent import AgentAction, LightMode
from app.schemas.emotion import EmotionStats


def decide_action(current_light_mode: LightMode, stats: EmotionStats) -> tuple[bool, AgentAction]:
    avg_score = stats.avg_emotion_score

    if current_light_mode == LightMode.normal_office_light:
        is_negative = stats.negative_ratio >= settings.negative_ratio_threshold
        score_is_low = avg_score is None or avg_score < settings.negative_score_threshold
        if stats.valid_count > 0 and is_negative and score_is_low:
            return True, AgentAction.switch_to_adjustment_light
        return False, AgentAction.keep_normal_office_light

    if current_light_mode == LightMode.adjustment_light:
        has_recovered = (
            stats.valid_count > 0
            and stats.negative_ratio <= settings.recovery_negative_ratio_threshold
            and avg_score is not None
            and avg_score >= settings.recovery_score_threshold
        )
        if has_recovered:
            return True, AgentAction.switch_to_normal_office_light
        return False, AgentAction.keep_adjustment_light

    return False, AgentAction.keep_normal_office_light

