from uuid import uuid4
import json
from pathlib import Path

from app.core.decision_engine import decide_action
from app.core.prompt_generator import build_strategy_message
from app.core.state_store import get_cooldown_remaining_seconds, is_in_cooldown
from app.db.crud import SessionModel
from app.schemas.agent import AgentAction, LightMode, StrategyResponse
from app.schemas.emotion import EmotionStats
from app.utils.time_utils import utc_now


def build_strategy(session: SessionModel, stats: EmotionStats) -> StrategyResponse:
    current_light_mode = LightMode(session.current_light_mode)
    trigger_popup, action = decide_action(current_light_mode, stats)
    in_cooldown = is_in_cooldown(session)
    cooldown_remaining = get_cooldown_remaining_seconds(session)

    if trigger_popup and in_cooldown:
        trigger_popup = False
        action = (
            AgentAction.keep_adjustment_light
            if current_light_mode == LightMode.adjustment_light
            else AgentAction.keep_normal_office_light
        )

    message = build_strategy_message(
        current_light_mode=current_light_mode,
        action=action,
        trigger_popup=trigger_popup,
        in_cooldown=in_cooldown,
    )

    recommended_lighting = None
    if trigger_popup:
        config_path = Path(__file__).resolve().parents[3] / "light_control" / "light_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        profile = "adjustment_light" if action == AgentAction.switch_to_adjustment_light else "normal_office_light"
        recommended_lighting = config["emotion_profiles"][profile]

    return StrategyResponse(
        strategy_id=f"strategy_{uuid4().hex[:12]}",
        session_id=session.session_id,
        timestamp=utc_now(),
        trigger_popup=trigger_popup,
        current_light_mode=current_light_mode,
        recommended_action=action,
        recommended_lighting=recommended_lighting,
        message=message,
        emotion_stats=stats,
        in_cooldown=in_cooldown,
        cooldown_remaining_seconds=cooldown_remaining,
    )

