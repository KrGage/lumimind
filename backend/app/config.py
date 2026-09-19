from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Lumimind Agent Backend"
    database_url: str = "sqlite:///./lumimind.db"

    demo_window_seconds: int = 60
    formal_window_seconds: int = 30 * 60

    demo_cooldown_seconds: int = 30
    formal_cooldown_seconds: int = 10 * 60

    negative_ratio_threshold: float = 0.6
    negative_score_threshold: float = 0.45
    recovery_negative_ratio_threshold: float = 0.3
    recovery_score_threshold: float = 0.55

    min_confidence: float = 0.6


settings = Settings()

