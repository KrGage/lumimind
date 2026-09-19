from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.schemas.agent import LightMode
from app.schemas.session import ExperimentMode, SessionInfo, SessionLogs, SessionStart

router = APIRouter(prefix="/api/session", tags=["session"])


@router.post("/start", response_model=SessionInfo)
def start_session(payload: SessionStart, db: Session = Depends(get_db)):
    return crud.upsert_session(db, payload)


@router.get("/logs", response_model=SessionLogs)
def get_session_logs(session_id: str, db: Session = Depends(get_db)):
    session = crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionLogs(
        session=SessionInfo(
            session_id=session.session_id,
            mode=ExperimentMode(session.mode),
            current_light_mode=LightMode(session.current_light_mode),
            started_at=session.started_at,
            updated_at=session.updated_at,
        ),
        emotions=[row_to_dict(item) for item in crud.list_all_emotions(db, session_id)],
        strategies=[row_to_dict(item) for item in crud.list_strategies(db, session_id)],
        light_mode_updates=[row_to_dict(item) for item in crud.list_light_mode_updates(db, session_id)],
        feedback=[row_to_dict(item) for item in crud.list_feedback(db, session_id)],
    )


def row_to_dict(row) -> dict:
    return {
        key: value
        for key, value in row.__dict__.items()
        if not key.startswith("_")
    }
