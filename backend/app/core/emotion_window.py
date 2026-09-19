from datetime import timedelta

from app.config import settings
from app.db.crud import EmotionModel
from app.schemas.emotion import EmotionLabel, EmotionStats
from app.schemas.session import ExperimentMode
from app.utils.time_utils import ensure_aware, utc_now


def get_window_seconds(mode: str) -> int:
    if mode == ExperimentMode.formal.value:
        return settings.formal_window_seconds
    return settings.demo_window_seconds


def get_window_start(mode: str):
    return utc_now() - timedelta(seconds=get_window_seconds(mode))


def calculate_emotion_stats(records: list[EmotionModel]) -> EmotionStats:
    total_count = len(records)
    valid_records = [
        item
        for item in records
        if item.face_detected and item.confidence >= settings.min_confidence
    ]
    valid_count = len(valid_records)
    if valid_count == 0:
        return EmotionStats(total_count=total_count, valid_count=0)

    positive_count = sum(1 for item in valid_records if item.emotion_label == EmotionLabel.positive.value)
    neutral_count = sum(1 for item in valid_records if item.emotion_label == EmotionLabel.neutral.value)
    negative_count = sum(1 for item in valid_records if item.emotion_label == EmotionLabel.negative.value)
    scored_records = [item for item in valid_records if item.emotion_score is not None]
    avg_score = None
    if scored_records:
        avg_score = round(sum(item.emotion_score or 0 for item in scored_records) / len(scored_records), 4)

    return EmotionStats(
        total_count=total_count,
        valid_count=valid_count,
        positive_count=positive_count,
        neutral_count=neutral_count,
        negative_count=negative_count,
        positive_ratio=round(positive_count / valid_count, 4),
        neutral_ratio=round(neutral_count / valid_count, 4),
        negative_ratio=round(negative_count / valid_count, 4),
        avg_emotion_score=avg_score,
        trend=calculate_trend(valid_records),
    )


def calculate_trend(records: list[EmotionModel]) -> str:
    scored = sorted(
        [item for item in records if item.emotion_score is not None],
        key=lambda item: ensure_aware(item.timestamp),
    )
    if len(scored) < 4:
        return "unknown"

    midpoint = len(scored) // 2
    earlier = scored[:midpoint]
    later = scored[midpoint:]
    earlier_avg = sum(item.emotion_score or 0 for item in earlier) / len(earlier)
    later_avg = sum(item.emotion_score or 0 for item in later) / len(later)
    delta = later_avg - earlier_avg

    if delta <= -0.08:
        return "worsening"
    if delta >= 0.08:
        return "recovering"
    return "stable"

