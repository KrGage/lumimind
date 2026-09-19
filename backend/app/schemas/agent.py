from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.emotion import EmotionStats


class LightMode(str, Enum):
    normal_office_light = "normal_office_light"
    adjustment_light = "adjustment_light"


class AgentAction(str, Enum):
    keep_normal_office_light = "keep_normal_office_light"
    switch_to_adjustment_light = "switch_to_adjustment_light"
    keep_adjustment_light = "keep_adjustment_light"
    switch_to_normal_office_light = "switch_to_normal_office_light"


class LightModeUpdate(BaseModel):
    session_id: str = Field(..., examples=["demo_001"])
    light_mode: LightMode
    operator_note: str | None = None


class StrategyResponse(BaseModel):
    strategy_id: str
    session_id: str
    timestamp: datetime
    trigger_popup: bool
    current_light_mode: LightMode
    recommended_action: AgentAction
    message: str
    emotion_stats: EmotionStats
    in_cooldown: bool = False
    cooldown_remaining_seconds: int = 0


class AgentCurrentState(BaseModel):
    session_id: str
    mode: str
    current_light_mode: LightMode
    recent_emotion_stats: EmotionStats
    last_strategy: StrategyResponse | None = None
    in_cooldown: bool
    cooldown_remaining_seconds: int

