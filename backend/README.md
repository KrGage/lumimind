# Lumimind Backend

Lumimind 是一个基于电脑摄像头情绪识别的办公光环境调节决策 Agent 后端。第一版只负责接收模型输出、统计情绪窗口、生成调光建议、记录人工执行和用户反馈，不直接控制硬件。

## 启动

```bash
cd D:\lumimind\backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

## 核心接口

- `POST /api/session/start` 创建或初始化 session
- `POST /api/model/emotion` 接收模型情绪识别结果并返回 Agent 策略
- `GET /api/agent/current?session_id=demo_001` 获取当前 Agent 状态
- `POST /api/agent/light-mode` 更新当前光模式
- `POST /api/feedback` 接收前端反馈
- `GET /api/session/logs?session_id=demo_001` 获取 session 日志

## 模式

- `demo`：统计最近 1 分钟情绪结果，冷却时间 30 秒
- `formal`：统计最近 30 分钟情绪结果，冷却时间 10 分钟

