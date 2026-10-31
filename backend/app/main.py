from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

import sys
from pathlib import Path

# 将 agent_framework 所在目录加入 sys.path，使 API 模块可以导入 agent_framework
_agent_root = Path(__file__).resolve().parent.parent.parent
if str(_agent_root) not in sys.path:
    sys.path.insert(0, str(_agent_root))

from app.api import agent_api, chat_api, feedback_api, library_api, manual_light_api, model_api, session_api
from app.config import settings
from app.db.database import init_db

app = FastAPI(title=settings.app_name)

@app.on_event("startup")
def on_startup() -> None:
    init_db()


# ── 前端页面路由映射表（集中管理，修改文件名只需改此处）──
PAGE_ROUTES = {
    "/page/home":     "/fontend/main.html",
    "/page/monitor":  "/fontend/index-V2.html",
    "/page/chat":     "/fontend/ai-chat.html",
    "/page/settings": "/fontend/settings.html",
    "/page/library":  "/fontend/library.html",
}


@app.get("/")
def root():
    """访问根路径直接跳转到前端主页面。"""
    return RedirectResponse(url="/page/home")


for _route, _target in PAGE_ROUTES.items():
    @app.get(_route, include_in_schema=False)
    def _page_redirect(target: str = _target):
        return RedirectResponse(url=target)


@app.get("/healthz")
def health_check():
    return {"status": "ok", "app": settings.app_name}


app.include_router(session_api.router)
app.include_router(model_api.router)
app.include_router(agent_api.router)
app.include_router(feedback_api.router)
app.include_router(chat_api.router)
app.include_router(manual_light_api.router)
app.include_router(library_api.router)

# 只公开前端文件；不要把 .env、数据库和模型文件作为静态资源暴露到局域网。
app.mount("/fontend", StaticFiles(directory=_agent_root / "fontend", html=True), name="frontend")

