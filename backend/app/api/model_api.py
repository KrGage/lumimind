from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.emotion_window import calculate_emotion_stats, get_window_start
from app.core.strategy_engine import build_strategy
from app.db import crud
from app.db.database import get_db
from app.schemas.agent import StrategyResponse
from app.schemas.emotion import EmotionCreate

router = APIRouter(prefix="/api/model", tags=["model"])


@router.post("/emotion", response_model=StrategyResponse)
def receive_emotion(payload: EmotionCreate, db: Session = Depends(get_db)):
    session = crud.get_session(db, payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Please start session first.")

    crud.add_emotion(db, payload)
    window_start = get_window_start(session.mode)
    records = crud.list_emotions_since(db, payload.session_id, window_start)
    stats = calculate_emotion_stats(records)
    strategy = build_strategy(session, stats)
    crud.save_strategy(db, strategy)
    return strategy

