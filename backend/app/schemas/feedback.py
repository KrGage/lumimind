from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class FeedbackLabel(str, Enum):
    better = "better"
    no_change = "no_change"
    worse = "worse"


class FeedbackCreate(BaseModel):
    session_id: str = Field(..., examples=["demo_001"])
    strategy_id: str
    accepted: bool
    executed: bool
    feedback_label: FeedbackLabel
    feedback_score: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = None


class FeedbackRecord(FeedbackCreate):
    id: int
    timestamp: datetime

    model_config = {"from_attributes": True}

