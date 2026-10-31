"""Bounded local history. Paths are resolved by integration, never against cwd."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


class Store:
    def __init__(self, path: Path, retention_days: int = 30):
        self.path = path
        self.retention_days = retention_days
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS samples (id INTEGER PRIMARY KEY, measured_at TEXT NOT NULL, payload TEXT NOT NULL)')
            db.execute('CREATE INDEX IF NOT EXISTS samples_time ON samples(measured_at)')

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.path), timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(self, sample: dict[str, Any]) -> None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO samples(measured_at,payload) VALUES (?,?)',
                       (sample['measured_at'], json.dumps(sample, allow_nan=False)))
            db.execute('DELETE FROM samples WHERE measured_at < ?', (cutoff,))

    def latest(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute('SELECT payload FROM samples ORDER BY measured_at DESC,id DESC LIMIT 1').fetchone()
        return json.loads(row[0]) if row else None

    def history(self, hours: float = 1, limit: int = 1000) -> list[dict[str, Any]]:
        if not 0 < hours <= 24 * 365 * 100 or not 1 <= limit <= 10000:
            raise ValueError('Invalid history window/limit')
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self.connect() as db:
            rows = db.execute('SELECT payload FROM samples WHERE measured_at >= ? ORDER BY measured_at DESC,id DESC LIMIT ?',
                              (cutoff, limit)).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]
