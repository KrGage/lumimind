from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.emotion_window import calculate_emotion_stats, get_window_start
from app.core.state_store import get_cooldown_remaining_seconds, is_in_cooldown
from app.db import crud
from app.db.database import get_db
from app.schemas.agent import AgentCurrentState, LightMode, LightModeUpdate, StrategyResponse
from app.schemas.emotion import EmotionStats

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.get("/current", response_model=AgentCurrentState)
def get_current_agent_state(session_id: str, db: Session = Depends(get_db)):
    session = crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    records = crud.list_emotions_since(db, session_id, get_window_start(session.mode))
    stats = calculate_emotion_stats(records)
    last_strategy = crud.get_last_strategy(db, session_id)

    return AgentCurrentState(
        session_id=session.session_id,
        mode=session.mode,
        current_light_mode=LightMode(session.current_light_mode),
        recent_emotion_stats=stats,
        last_strategy=to_strategy_response(last_strategy) if last_strategy else None,
        in_cooldown=is_in_cooldown(session),
        cooldown_remaining_seconds=get_cooldown_remaining_seconds(session),
    )


@router.post("/light-mode")
def update_light_mode(payload: LightModeUpdate, db: Session = Depends(get_db)):
    try:
        update = crud.update_light_mode(
            db=db,
            session_id=payload.session_id,
            light_mode=payload.light_mode,
            operator_note=payload.operator_note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "session_id": update.session_id,
        "light_mode": update.light_mode,
        "timestamp": update.timestamp,
        "operator_note": update.operator_note,
    }


def to_strategy_response(model) -> StrategyResponse:
    return StrategyResponse(
        strategy_id=model.strategy_id,
        session_id=model.session_id,
        timestamp=model.timestamp,
        trigger_popup=model.trigger_popup,
        current_light_mode=LightMode(model.current_light_mode),
        recommended_action=model.recommended_action,
        message=model.message,
        emotion_stats=EmotionStats(
            total_count=model.total_count,
            valid_count=model.valid_count,
            negative_ratio=model.negative_ratio,
            avg_emotion_score=model.avg_emotion_score,
            trend=model.trend,
        ),
    )

