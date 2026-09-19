from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.agent import LightMode


class ExperimentMode(str, Enum):
    demo = "demo"
    formal = "formal"


class SessionStart(BaseModel):
    session_id: str = Field(..., examples=["demo_001"])
    mode: ExperimentMode = ExperimentMode.demo
    initial_light_mode: LightMode = LightMode.normal_office_light


class SessionInfo(BaseModel):
    session_id: str
    mode: ExperimentMode
    current_light_mode: LightMode
    started_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SessionLogs(BaseModel):
    session: SessionInfo
    emotions: list[dict]
    strategies: list[dict]
    light_mode_updates: list[dict]
    feedback: list[dict]

