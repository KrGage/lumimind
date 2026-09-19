from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.database import get_db
from app.schemas.feedback import FeedbackCreate, FeedbackRecord

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackRecord)
def receive_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    if crud.get_session(db, payload.session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return crud.add_feedback(db, payload)

