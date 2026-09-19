from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EmotionLabel(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class EmotionCreate(BaseModel):
    session_id: str = Field(..., examples=["demo_001"])
    timestamp: datetime
    emotion_label: EmotionLabel
    emotion_score: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    face_detected: bool = True


class EmotionRecord(EmotionCreate):
    id: int

    model_config = {"from_attributes": True}


class EmotionStats(BaseModel):
    total_count: int = 0
    valid_count: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    positive_ratio: float = 0.0
    neutral_ratio: float = 0.0
    negative_ratio: float = 0.0
    avg_emotion_score: float | None = None
    trend: str = "unknown"

