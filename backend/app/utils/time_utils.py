from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def seconds_between(later: datetime, earlier: datetime) -> int:
    return max(0, int((ensure_aware(later) - ensure_aware(earlier)).total_seconds()))

