from app.config import settings
from app.db.crud import SessionModel
from app.schemas.session import ExperimentMode
from app.utils.time_utils import seconds_between, utc_now


def get_cooldown_seconds(mode: str) -> int:
    if mode == ExperimentMode.formal.value:
        return settings.formal_cooldown_seconds
    return settings.demo_cooldown_seconds


def get_cooldown_remaining_seconds(session: SessionModel) -> int:
    if session.last_popup_at is None:
        return 0

    elapsed = seconds_between(utc_now(), session.last_popup_at)
    return max(0, get_cooldown_seconds(session.mode) - elapsed)


def is_in_cooldown(session: SessionModel) -> bool:
    return get_cooldown_remaining_seconds(session) > 0

