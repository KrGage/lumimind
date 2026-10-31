"""register_ultra(app) before a catch-all static mount. One uvicorn worker only."""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json
from pathlib import Path
import re
from typing import Any, AsyncIterator

from fastapi import APIRouter, FastAPI, Query
from fastapi.responses import FileResponse

from .collector import Collector
from .lighting import lighting_service
from .storage import Store

ROOT = Path(__file__).resolve().parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    config = json.loads((path or ROOT / 'config.json').read_text(encoding='utf-8-sig'))
    if type(config.get('enabled')) is not bool:
        raise ValueError('enabled 必须为布尔值')
    if not re.fullmatch(r'(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}', str(config.get('ble_address', ''))):
        raise ValueError('ble_address 必须为 BLE MAC 地址')
    if not re.fullmatch(r'ULTRA-[A-Za-z0-9_-]+', str(config.get('device_id', ''))):
        raise ValueError('设备编号必须以 ULTRA- 开头')
    for key, lower, upper in [('interval_seconds', 12, 300), ('retry_seconds', 1, 300), ('retention_days', 1, 3650)]:
        value = config.get(key)
        if type(value) not in (int, float) or not lower <= value <= upper:
            raise ValueError(f'{key} 必须在 {lower}..{upper} 之间')
    database = config.get('database')
    if not isinstance(database, str) or not database.strip():
        raise ValueError('database 必须为数据库文件路径')
    return config


def register_ultra(app: FastAPI, *, config_path: Path | None = None) -> None:
    if getattr(app.state, 'ultra_registered', False):
        return
    config = load_config(config_path)
    base = config_path.resolve().parent if config_path else ROOT
    store = Store(base / config['database'], int(config['retention_days']))
    collector = Collector(config, store)
    app.state.ultra_registered = True
    app.state.ultra_collector = collector
    previous_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[Any]:
        async with previous_lifespan(application) as state:
            await collector.start()
            try:
                yield state
            finally:
                await collector.stop()

    app.router.lifespan_context = lifespan
    router = APIRouter(prefix='/ultra', tags=['ultra'])

    @router.get('/dashboard', include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(ROOT / 'static' / 'dashboard.html')

    @router.get('/api/latest')
    def latest() -> dict[str, Any]:
        sample = store.latest()
        age = None
        if sample:
            age = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(sample['measured_at'])).total_seconds())
        return {'connection': dict(collector.status), 'sample': sample, 'age_seconds': age,
                'stale': age is None or age > config['interval_seconds'] * 3,
                'enabled': config['enabled'], 'lighting': lighting_service.last}

    @router.get('/api/history')
    def history(hours: float = Query(1, gt=0, le=720), limit: int = Query(1000, ge=1, le=10000)) -> dict[str, Any]:
        return {'samples': store.history(hours, limit)}

    app.include_router(router)
