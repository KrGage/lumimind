# ai_agent — 欧普照明健康光环境智能体

面向"光源智能体设计大赛"的 AI 对话与健康光环境推荐智能体。基于 LLM 多模型降级链 + RAG 知识库 + 情绪/生理状态分析，为用户提供健康照明建议与 AI 对话服务。

## 目录结构

```
ai_agent/
├── agent_framework/        # 核心 Agent 框架
│   ├── core_ai.py          #   多模型 AI 核心（含 Gemini 等多后端接入与降级链）
│   ├── llm_wrapper.py      #   LLM 统一封装（重试、容错、降级）
│   ├── guardrails.py       #   输入/输出安全护栏
│   ├── workflow.py         #   Agent 工作流（LangGraph 状态流转）
│   ├── rag.py              #   RAG 检索问答
│   ├── rag_pipeline.py     #   RAG 管道（检索 + 生成）
│   ├── semantic_cache.py   #   语义缓存
│   ├── tools.py            #   Agent 可调用的工具集
│   ├── prompt_registry.py  #   提示词注册中心
│   └── prompts/            #   框架内置提示词
├── backend/                # FastAPI 后端服务
│   └── app/
│       ├── main.py         #   应用入口（uvicorn app.main:app 启动）
│       ├── config.py       #   配置
│       ├── api/            #   路由：chat、agent、session、model、feedback
│       ├── core/           #   业务核心：决策引擎、情绪窗口、策略引擎、提示词生成、状态存储
│       ├── db/             #   数据库（SQLite + SQLAlchemy CRUD）
│       ├── schemas/        #   Pydantic 模型
│       └── utils/          #   工具函数
├── fontend/                # 前端页面（纯 HTML）
│   ├── ai-chat.html        #   AI 对话页面
│   ├── ai-chat2.html       #   AI 对话页面（V2）
│   ├── main.html           #   主页面
│   ├── index-V2.html       #   主页面（V2）
│   └── settings.html       #   设置页面
├── prompts/                # 项目提示词
│   ├── prompt_texts.py     #   提示词文本（医疗/照明专用提示词体系）
│   └── registry_init.py    #   注册中心初始化
├── custom_agents/          # 自定义 Agent 扩展（example_agent.py 示例）
├── custom_prompts/         # 自定义提示词扩展（example_prompts.py 示例）
├── Oupu_Lighting/          # 欧普照明子项目：数据分析与实验脚本
│   ├── scripts/            #   ECG 数据处理、情绪识别模型、光谱绘制等脚本
│   ├── data/               #   数据（All_Data.xlsx、ecg_data.pkl、C-data 等）
│   └── outputs/            #   输出结果（工作室光谱图等）
├── models/                 # 本地模型产物（vector_store.json 向量库、semantic_cache.json 语义缓存）
├── ingest_knowledge_base.py  # 知识库构建脚本（PDF/DOCX 解析入库）
├── oupu_lighting_agent.py    # 智能体主入口脚本
├── requirements.txt          # Python 依赖清单
├── .env                      # 环境变量（模型 API Key、URL 等，勿提交）
└── 教程.md                   # 使用教程
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 .env（模型 API Key 等，参考现有 .env 结构）

# 3. 启动后端服务（在 backend/ 目录下）
cd backend
uvicorn app.main:app --reload

# 4. 打开前端页面
#    直接用浏览器打开 fontend/main.html 或 ai-chat.html

# 5. 构建知识库（首次使用 RAG 前）
python ingest_knowledge_base.py

# 6. 运行智能体主入口
python oupu_lighting_agent.py
```

## 模块说明

| 模块 | 职责 |
|------|------|
| `agent_framework` | AI 能力核心：多模型降级链（Qwen/Gemini 等）、RAG 问答、语义缓存、护栏、工作流 |
| `backend` | FastAPI 服务，提供 AI 对话、会话管理、模型选择、反馈收集等 API，含情绪决策与策略引擎 |
| `fontend` | 静态 HTML 前端，含 AI 对话与系统主页面（注：目录名沿用历史拼写 fontend） |
| `prompts` / `custom_prompts` | 医疗/照明专用提示词体系，支持扩展注册 |
| `Oupu_Lighting` | 生理信号（ECG/HRV）情绪识别实验脚本与工作室光谱分析 |
| `models` | RAG 向量库与语义缓存的本地持久化产物 |

## 注意事项

- `.env` 包含 API 密钥，请勿提交到版本库。
- `backend/lumimind.db` 为 SQLite 数据库文件，运行时自动生成。
- `Oupu_Lighting/scripts` 中的深度学习脚本依赖 `torch`，仅训练/推理时需要。
