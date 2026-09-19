from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.db.database import Base
from app.schemas.agent import AgentAction, LightMode, StrategyResponse
from app.schemas.emotion import EmotionCreate
from app.schemas.feedback import FeedbackCreate
from app.schemas.session import ExperimentMode, SessionStart
from app.utils.time_utils import utc_now


class SessionModel(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    mode: Mapped[str] = mapped_column(String, default=ExperimentMode.demo.value)
    current_light_mode: Mapped[str] = mapped_column(String, default=LightMode.normal_office_light.value)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_popup_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EmotionModel(Base):
    __tablename__ = "emotions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    emotion_label: Mapped[str] = mapped_column(String)
    emotion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    face_detected: Mapped[bool] = mapped_column(Boolean, default=True)


class StrategyModel(Base):
    __tablename__ = "strategies"

    strategy_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    trigger_popup: Mapped[bool] = mapped_column(Boolean, default=False)
    current_light_mode: Mapped[str] = mapped_column(String)
    recommended_action: Mapped[str] = mapped_column(String)
    message: Mapped[str] = mapped_column(Text)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    valid_count: Mapped[int] = mapped_column(Integer, default=0)
    negative_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    avg_emotion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    trend: Mapped[str] = mapped_column(String, default="unknown")


class LightModeUpdateModel(Base):
    __tablename__ = "light_mode_updates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    light_mode: Mapped[str] = mapped_column(String)
    operator_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class FeedbackModel(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.session_id"), index=True)
    strategy_id: Mapped[str] = mapped_column(String, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    accepted: Mapped[bool] = mapped_column(Boolean)
    executed: Mapped[bool] = mapped_column(Boolean)
    feedback_label: Mapped[str] = mapped_column(String)
    feedback_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)


def upsert_session(db: Session, payload: SessionStart) -> SessionModel:
    session = db.get(SessionModel, payload.session_id)
    now = utc_now()
    if session is None:
        session = SessionModel(
            session_id=payload.session_id,
            mode=payload.mode.value,
            current_light_mode=payload.initial_light_mode.value,
            started_at=now,
            updated_at=now,
        )
        db.add(session)
    else:
        session.mode = payload.mode.value
        session.current_light_mode = payload.initial_light_mode.value
        session.updated_at = now
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: str) -> SessionModel | None:
    return db.get(SessionModel, session_id)


def require_session(db: Session, session_id: str) -> SessionModel:
    session = get_session(db, session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")
    return session


def add_emotion(db: Session, payload: EmotionCreate) -> EmotionModel:
    emotion = EmotionModel(**payload.model_dump(mode="python"))
    emotion.emotion_label = payload.emotion_label.value
    db.add(emotion)
    db.commit()
    db.refresh(emotion)
    return emotion


def list_emotions_since(db: Session, session_id: str, since: datetime) -> list[EmotionModel]:
    statement = (
        select(EmotionModel)
        .where(EmotionModel.session_id == session_id, EmotionModel.timestamp >= since)
        .order_by(EmotionModel.timestamp.asc())
    )
    return list(db.scalars(statement))


def list_all_emotions(db: Session, session_id: str) -> list[EmotionModel]:
    return list(db.scalars(select(EmotionModel).where(EmotionModel.session_id == session_id)))


def save_strategy(db: Session, strategy: StrategyResponse) -> StrategyModel:
    stats = strategy.emotion_stats
    model = StrategyModel(
        strategy_id=strategy.strategy_id,
        session_id=strategy.session_id,
        timestamp=strategy.timestamp,
        trigger_popup=strategy.trigger_popup,
        current_light_mode=strategy.current_light_mode.value,
        recommended_action=strategy.recommended_action.value,
        message=strategy.message,
        total_count=stats.total_count,
        valid_count=stats.valid_count,
        negative_ratio=stats.negative_ratio,
        avg_emotion_score=stats.avg_emotion_score,
        trend=stats.trend,
    )
    db.add(model)
    if strategy.trigger_popup:
        session = require_session(db, strategy.session_id)
        session.last_popup_at = strategy.timestamp
        session.updated_at = strategy.timestamp
    db.commit()
    db.refresh(model)
    return model


def get_last_strategy(db: Session, session_id: str) -> StrategyModel | None:
    statement = (
        select(StrategyModel)
        .where(StrategyModel.session_id == session_id)
        .order_by(StrategyModel.timestamp.desc())
        .limit(1)
    )
    return db.scalar(statement)


def update_light_mode(db: Session, session_id: str, light_mode: LightMode, operator_note: str | None) -> LightModeUpdateModel:
    session = require_session(db, session_id)
    now = utc_now()
    session.current_light_mode = light_mode.value
    session.updated_at = now
    update = LightModeUpdateModel(
        session_id=session_id,
        timestamp=now,
        light_mode=light_mode.value,
        operator_note=operator_note,
    )
    db.add(update)
    db.commit()
    db.refresh(update)
    return update


def add_feedback(db: Session, payload: FeedbackCreate) -> FeedbackModel:
    feedback = FeedbackModel(**payload.model_dump(mode="python"))
    feedback.feedback_label = payload.feedback_label.value
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def list_strategies(db: Session, session_id: str) -> list[StrategyModel]:
    return list(db.scalars(select(StrategyModel).where(StrategyModel.session_id == session_id)))


def list_light_mode_updates(db: Session, session_id: str) -> list[LightModeUpdateModel]:
    return list(db.scalars(select(LightModeUpdateModel).where(LightModeUpdateModel.session_id == session_id)))


def list_feedback(db: Session, session_id: str) -> list[FeedbackModel]:
    return list(db.scalars(select(FeedbackModel).where(FeedbackModel.session_id == session_id)))

