"""Manual light targets are sent once through the local SDK adapter."""

import json
import ipaddress
from threading import Lock
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.crud import ManualLightCommandModel
from app.db.database import get_db
from app.schemas.manual_light import ManualLightRequest
from app.utils.time_utils import utc_now
from light_control.lumimind_light import LightingDeliveryUnknown, LightingError, set_light


router = APIRouter(prefix="/api/light/manual", tags=["manual-light"])
_SEND_LOCK = Lock()


def _require_local_request(request: Request) -> None:
    """The hardware endpoint is for the page running on this computer."""
    peer = request.client.host if request.client else ""
    try:
        local_peer = ipaddress.ip_address(peer).is_loopback
    except ValueError:
        local_peer = False
    if not local_peer:
        raise HTTPException(403, "手动灯控只允许本机访问。")
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        try:
            origin_port = parsed.port
        except ValueError as exc:
            raise HTTPException(403, "请从本机 agent 页面操作灯具。") from exc
        if (parsed.scheme != request.url.scheme
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or origin_port != request.url.port):
            raise HTTPException(403, "请从本机 agent 页面操作灯具。")


def _response(row: ManualLightCommandModel) -> dict:
    result = json.loads(row.result_json) if row.result_json else None
    return {
        "command_id": row.command_id,
        "session_id": row.session_id,
        "target": {"cct_k": row.cct_k, "illuminance_lux": row.illuminance_lux},
        "status": row.status,
        "hardware_confirmed": False,
        "execution": result,
        "error": row.error,
        "created_at": row.created_at,
        "completed_at": row.completed_at,
    }


@router.get("/latest")
def latest_manual_light(db: Session = Depends(get_db)):
    row = db.scalar(
        select(ManualLightCommandModel)
        .order_by(ManualLightCommandModel.created_at.desc())
        .limit(1)
    )
    return _response(row) if row else None


@router.post("")
def apply_manual_light(payload: ManualLightRequest, request: Request, db: Session = Depends(get_db)):
    _require_local_request(request)
    command_id = str(payload.command_id)
    with _SEND_LOCK:
        row = db.get(ManualLightCommandModel, command_id)
        if row is not None:
            if (row.cct_k != payload.cct_k
                    or row.illuminance_lux != payload.illuminance_lux
                    or row.session_id != payload.session_id):
                raise HTTPException(409, "该命令编号对应另一组参数。")
            if row.status == "sent":
                return _response(row)
            raise HTTPException(409, f"命令已处于 {row.status}，不会自动重复发送；请核对灯具和记录。")

        row = ManualLightCommandModel(
            command_id=command_id,
            session_id=payload.session_id,
            cct_k=payload.cct_k,
            illuminance_lux=payload.illuminance_lux,
            status="dispatching",
            created_at=utc_now(),
        )
        db.add(row)
        db.commit()  # Keep the command ID before contacting the lamp.

        try:
            result = set_light(payload.illuminance_lux, payload.cct_k)
            if result.get("status") != "sent":
                raise LightingDeliveryUnknown("灯控模块未确认报文已发送。")
            result_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
        except LightingDeliveryUnknown as exc:
            row.status = "unknown"
            row.error = str(exc)
        except LightingError as exc:
            row.status = "failed"
            row.error = str(exc)
        except Exception as exc:
            row.status = "unknown"
            row.error = str(exc)
        else:
            row.status = "sent"
            row.result_json = result_json
        row.completed_at = utc_now()

        try:
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            raise HTTPException(503, "指令可能已发送，但执行记录保存失败；请核对灯具，勿直接重发。") from exc

        response = _response(row)
        if row.status != "sent":
            raise HTTPException(503, detail=response)
        return response
