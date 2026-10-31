"""Standalone dashboard server: real-time ULTRA data, no full agent needed."""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / 'payload'))
sys.path.insert(0, str(_root / 'payload' / 'backend'))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from ultra_monitor.integration import register_ultra  # noqa: E402

app = FastAPI(title='ULTRA Dashboard Standalone')


@app.get('/')
def root():
    return RedirectResponse(url='/ultra/dashboard')


register_ultra(app)
