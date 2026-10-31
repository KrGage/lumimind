"""Accept a saved emotion recommendation and dispatch its exact numeric target once."""

import json
from threading import Lock
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.lighting_profiles import TARGET_MODES
from app.db import crud
from app.schemas.agent import AgentAction
from app.schemas.lighting import LightingTarget
from app.utils.time_utils import utc_now
from light_control.lumimind_light import LightingError, set_light

_LAMP_LOCK = Lock()


class LightingRequestError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def saved_target(row: crud.StrategyLightingModel) -> LightingTarget:
    return LightingTarget(cct_k=row.cct_k, illuminance_lux=row.illuminance_lux)


def current_lighting(db: Session, session_id: str) -> LightingTarget | None:
    row = db.scalar(
        select(crud.StrategyLightingModel)
        .join(crud.StrategyModel)
        .where(crud.StrategyModel.session_id == session_id, crud.StrategyLightingModel.status == "sent")
        .order_by(crud.StrategyLightingModel.completed_at.desc())
        .limit(1)
    )
    return saved_target(row) if row is not None else None


def _response(model: crud.StrategyModel, row: crud.StrategyLightingModel, already_sent: bool) -> dict[str, Any]:
    return {
        "session_id": model.session_id, "strategy_id": model.strategy_id,
        "current_light_mode": TARGET_MODES[AgentAction(model.recommended_action)].value,
        "recommended_lighting": saved_target(row).model_dump(),
        "execution": json.loads(row.result_json or "{}"),
        "already_sent": already_sent,
    }


def accept_lighting(db: Session, session_id: str, strategy_id: str) -> dict[str, Any]:
    # Serializes the single physical lamp in this backend process. The committed
    # pending->dispatching database transition also prevents duplicate sends
    # across concurrent workers or after restarting the backend.
    with _LAMP_LOCK:
        db.expire_all()
        session = crud.get_session(db, session_id)
        model = db.get(crud.StrategyModel, strategy_id)
        if session is None or model is None or model.session_id != session_id:
            raise LightingRequestError(404, "找不到该会话的调光建议。")
        row = model.lighting
        if row is None:
            raise LightingRequestError(409, "这条建议没有明确的照度和色温，请生成新的建议。")
        if row.status == "sent":
            return _response(model, row, True)
        if row.status != "pending":
            raise LightingRequestError(409, f"这条建议状态为 {row.status}，不会重复下发，请核对执行记录。")

        mode = TARGET_MODES.get(AgentAction(model.recommended_action))
        latest = db.scalar(
            select(crud.StrategyModel.strategy_id)
            .where(crud.StrategyModel.session_id == session_id, crud.StrategyModel.trigger_popup.is_(True))
            .order_by(crud.StrategyModel.timestamp.desc()).limit(1)
        )
        if mode is None or not model.trigger_popup or latest != strategy_id or session.current_light_mode != model.current_light_mode:
            raise LightingRequestError(409, "建议已过期或光模式已变化，请使用最新建议。")
        target = saved_target(row)
        claimed = cast(CursorResult[Any], db.execute(
            update(crud.StrategyLightingModel)
            .where(crud.StrategyLightingModel.strategy_id == strategy_id, crud.StrategyLightingModel.status == "pending")
            .values(status="dispatching")
        ))
        if claimed.rowcount != 1:
            db.rollback()
            raise LightingRequestError(409, "这条建议已由其他请求处理，不重复下发。")
        db.commit()  # Must survive process interruption BEFORE contacting hardware.

        try:
            result = set_light(illuminance_lux=target.illuminance_lux, cct_k=target.cct_k)
            if result.get("status") != "sent":
                raise RuntimeError("控制模块未返回 sent 状态。")
        except Exception as exc:
            row.status = "failed" if isinstance(exc, LightingError) else "unknown"
            row.error = str(exc)
            row.completed_at = utc_now()
            db.commit()
            raise LightingRequestError(503, f"自动调光未完成：{exc}。请核对配置和灯具；本条建议不会自动重发。") from exc

        try:
            row.status = "sent"
            row.result_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
            row.error = None
            row.completed_at = utc_now()
            session.current_light_mode = mode.value
            session.updated_at = row.completed_at
            db.add(crud.LightModeUpdateModel(
                session_id=session_id, timestamp=row.completed_at, light_mode=mode.value,
                operator_note=f"接受建议 {strategy_id} 后自动下发 {target.cct_k}K / {target.illuminance_lux:g}lx；硬件未回执。",
            ))
            db.commit()
        except (SQLAlchemyError, ValueError) as exc:
            db.rollback()
            raise LightingRequestError(503, "指令可能已发送，但执行结果未能保存；请核对灯具，不要重复发送。") from exc
        return _response(model, row, False)


def reject_lighting(db: Session, session_id: str, strategy_id: str) -> None:
    with _LAMP_LOCK:
        db.expire_all()
        model = db.get(crud.StrategyModel, strategy_id)
        if model is None or model.session_id != session_id:
            raise LightingRequestError(404, "找不到该会话的建议。")
        row = model.lighting
        if row is None or row.status == "rejected":
            return
        if row.status != "pending":
            raise LightingRequestError(409, "该建议已进入执行流程，不能再标为未接受。")
        result = cast(CursorResult[Any], db.execute(
            update(crud.StrategyLightingModel)
            .where(crud.StrategyLightingModel.strategy_id == strategy_id, crud.StrategyLightingModel.status == "pending")
            .values(status="rejected", completed_at=utc_now())
        ))
        if result.rowcount != 1:
            db.rollback()
            raise LightingRequestError(409, "建议已被其他请求处理。")
        db.commit()
