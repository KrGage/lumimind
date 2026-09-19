# Lumimind 后端对接说明

用于前端、模型和后端对齐接口与分工。

## 1. 项目位置

后端代码已经保存在：

```text
D:\lumimind\backend
```

后端基于 FastAPI 搭建，目前只负责：

- 接收情绪识别模型输出
- 统计滑动时间窗口内的情绪状态
- 根据当前光模式生成调光建议
- 接收前端反馈
- 记录 session、情绪结果、Agent 策略、光模式更新和用户反馈日志

后端不直接控制 cube 硬件。cube 的真实切换由实验员人工完成，前端负责把人工切换后的当前光模式通知后端。

## 2. 启动方式

在本机启动后端：

```powershell
cd D:\lumimind\backend
.\.venv\Scripts\activate
uvicorn app.main:app --reload
```

后端基础地址：

```text
http://127.0.0.1:8000
```

Swagger 接口文档：

```text
http://127.0.0.1:8000/docs
```

## 3. 前端需要知道的信息

### 3.1 创建或初始化 session

前端在一次实验或演示开始前调用：

```http
POST /api/session/start
```

请求示例：

```json
{
  "session_id": "demo_001",
  "mode": "demo",
  "initial_light_mode": "normal_office_light"
}
```

字段说明：

| 字段 | 含义 | 可选值 |
| --- | --- | --- |
| `session_id` | 当前实验或演示编号 | 自定义字符串，例如 `demo_001` |
| `mode` | 实验模式 | `demo` / `formal` |
| `initial_light_mode` | 初始光模式 | `normal_office_light` / `adjustment_light` |

### 3.2 获取当前 Agent 状态

前端可以按需轮询或在页面刷新时调用：

```http
GET /api/agent/current?session_id=demo_001
```

返回中重点关注：

```json
{
  "session_id": "demo_001",
  "mode": "demo",
  "current_light_mode": "normal_office_light",
  "recent_emotion_stats": {},
  "last_strategy": {},
  "in_cooldown": false,
  "cooldown_remaining_seconds": 0
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `current_light_mode` | 后端记录的当前光模式 |
| `recent_emotion_stats` | 最近时间窗口内的情绪统计 |
| `last_strategy` | 最近一次 Agent 策略 |
| `in_cooldown` | 是否处于冷却时间 |
| `cooldown_remaining_seconds` | 冷却剩余秒数 |

### 3.3 根据后端策略弹窗

模型提交情绪结果后，后端会返回 Agent 策略。前端重点看：

```json
{
  "strategy_id": "strategy_xxx",
  "trigger_popup": true,
  "current_light_mode": "normal_office_light",
  "recommended_action": "switch_to_adjustment_light",
  "message": "检测到近一段时间情绪状态偏低，是否切换至光辅助调节模式？",
  "emotion_stats": {}
}
```

如果：

```text
trigger_popup = true
```

前端就弹出 `message`。

如果：

```text
trigger_popup = false
```

前端不弹窗，只展示当前状态即可。

### 3.4 更新当前光模式

当实验员手动调完 cube 后，前端需要通知后端当前光模式：

```http
POST /api/agent/light-mode
```

请求示例：

```json
{
  "session_id": "demo_001",
  "light_mode": "adjustment_light",
  "operator_note": "实验员已手动将 cube 调整至调节光"
}
```

光模式只支持：

```text
normal_office_light
adjustment_light
```

### 3.5 提交用户或实验员反馈

前端在用户/实验员对建议做出反馈后调用：

```http
POST /api/feedback
```

请求示例：

```json
{
  "session_id": "demo_001",
  "strategy_id": "strategy_xxx",
  "accepted": true,
  "executed": true,
  "feedback_label": "better",
  "feedback_score": 4,
  "comment": "调节后感觉有所缓解"
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `strategy_id` | 后端策略 ID，用于追踪是哪次建议 |
| `accepted` | 用户或实验员是否接受建议 |
| `executed` | 是否已经实际执行 cube 调节 |
| `feedback_label` | 调节后的主观反馈 |
| `feedback_score` | 1 到 5 分，可选 |
| `comment` | 备注，可选 |

反馈标签只支持：

```text
better
no_change
worse
```

## 4. 模型同学需要知道的信息

### 4.1 提交情绪识别结果

模型同学每次识别后调用：

```http
POST /api/model/emotion
```

请求示例：

```json
{
  "session_id": "demo_001",
  "timestamp": "2026-05-19T10:00:00",
  "emotion_label": "negative",
  "emotion_score": 0.32,
  "confidence": 0.86,
  "face_detected": true
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `session_id` | 当前实验/演示编号，必须和前端创建的 session_id 一致 |
| `timestamp` | 识别时间，ISO 格式 |
| `emotion_label` | 情绪标签 |
| `emotion_score` | 情绪分数，范围 0.0 到 1.0 |
| `confidence` | 模型置信度，范围 0.0 到 1.0 |
| `face_detected` | 是否检测到人脸 |

情绪标签只支持：

```text
positive
neutral
negative
```

情绪分数约定：

```text
0.0 = 非常消极
0.5 = 中性
1.0 = 非常积极
```

当前后端会忽略以下结果：

```text
confidence < 0.6
face_detected = false
```

因此模型侧最好稳定传入 `confidence` 和 `face_detected`。

## 5. 当前后端决策规则

### 5.1 正常办公光下

如果当前光模式是：

```text
normal_office_light
```

并且最近时间窗口内：

```text
negative_ratio >= 0.6
avg_emotion_score < 0.45
```

则后端建议切换到调节光：

```text
switch_to_adjustment_light
```

并触发弹窗。

### 5.2 调节光下

如果当前光模式是：

```text
adjustment_light
```

并且最近时间窗口内：

```text
negative_ratio <= 0.3
avg_emotion_score >= 0.55
```

则后端建议切回正常办公光：

```text
switch_to_normal_office_light
```

并触发弹窗。

### 5.3 调节光下仍然消极

如果当前已经是调节光，但情绪仍偏消极：

```text
recommended_action = keep_adjustment_light
trigger_popup = false
```

后端不会反复弹窗。

### 5.4 冷却时间

为了避免频繁弹窗，后端加入冷却机制：

| 模式 | 情绪统计窗口 | 冷却时间 |
| --- | --- | --- |
| `demo` | 最近 1 分钟 | 30 秒 |
| `formal` | 最近 30 分钟 | 10 分钟 |

## 6. 建议确认的问题

### 6.1 需要和前端确认

- 前端是否负责在实验开始前调用 `POST /api/session/start`
- 前端弹窗按钮如何设计：接受、拒绝、是否已执行 cube 调节
- 实验员手动调完 cube 后，由谁点击“已切换光模式”
- 前端是否需要轮询 `GET /api/agent/current`
- 前端是否展示最近情绪统计，例如 `negative_ratio`、`avg_emotion_score`、`trend`

### 6.2 需要和模型确认

- 模型能否输出 `positive`、`neutral`、`negative` 三类标签
- 模型能否输出 `emotion_score`
- demo 模式下模型识别频率是否按 10 秒一次
- 模型是否直接请求后端，还是先发给前端再由前端转发
- `timestamp` 由模型生成，还是由后端接收时生成


