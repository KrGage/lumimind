# Lumimind Backend

Lumimind 后端接收情绪结果、生成建议并记录反馈。主界面另提供独立的手动单灯控制接口：明确输入照度和色温并点击下发后，调用本地 SDK 配方计算与 UDP 灯控；情绪建议仍按原有流程运行。

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
- `POST /api/light/manual` 手动下发照度与色温（本机页面使用，需唯一 `command_id`）
- `GET /api/light/manual/latest` 查看最近一条手动命令与发送状态

手动调光需要在 `ai_agent/light_control/light_config.json` 配置真实灯具 IP，并核对随包标定文件。`sent` 只表示 UDP 报文已发送，不代表实测达标；参数与运行条件见 `ai_agent/light_control/README.md`。灯控只从本机访问，后端使用一个 worker。

情绪监测页的调光弹窗提供“使用推荐参数”“自定义参数”“暂不接受”。前两项都调用同一个单灯手动调光接口，每次确认生成新的命令编号；暂不接受只记录反馈。当前本地情绪策略只决定光模式，弹窗数值来自 `light_config.json` 的 `emotion_profiles` 演示预设，不代表按情绪实时优化的处方。发送成功后仍须用外部监测核对实际照度与色温；发送结果未确认时不要直接重试。

## 模式

- `demo`：统计最近 1 分钟情绪结果，冷却时间 30 秒
- `formal`：统计最近 30 分钟情绪结果，冷却时间 10 分钟

