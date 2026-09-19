"""
Agent Framework — Multi-Tier AI Inference Engine
=================================================

所有 AI 推理必须通过此模块。绝不直接调用 Provider API。

三层自动降级:
  Tier A: Ollama（本地优先，无需云服务）
  Tier B: Gemini（Google API，有免费额度）
  Tier C: OpenAI / Anthropic / OpenRouter（可选，通过环境变量或代码参数）

用法:
    from agent_framework.core_ai import generate, chat, chat_stream, is_available

    text = await generate("Hello, how are you?", system="You are a helpful assistant.")
    text = await chat([{"role": "user", "content": "What is AI?"}], system="...")
    async for chunk in chat_stream(messages, system="..."):
        print(chunk, end="")

从 AI-Healthcare-System 抽取而来，已移除医疗领域特化逻辑。
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, List, Optional

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


# ── 工具函数 ─────────────────────────────────────────────────────
def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _resolve_import(module_path: str, name: str):
    """延迟导入，避免循环依赖。"""
    import importlib
    return getattr(importlib.import_module(module_path), name, None)


# ── 配置 ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = _env_int("OLLAMA_TIMEOUT", 120)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", GEMINI_MODEL)

# DeepSeek（OpenAI 兼容 API）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 通义千问 Qwen / DashScope（OpenAI 兼容 API）
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

# 智谱 GLM / BigModel（OpenAI 兼容 API）
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4-flash")

# 模型列表缓存（避免冗余 /api/tags 调用）
_model_cache: dict[str, tuple[float, list[str]]] = {}
_MODEL_CACHE_TTL = 30  # 秒


# ═════════════════════════════════════════════════════════════════
# TIER A: OLLAMA（本地推理）
# ═════════════════════════════════════════════════════════════════

async def get_ollama_models() -> list[str]:
    """列出可用的 Ollama 模型（TTL 缓存）。"""
    import time as _time
    cache_key = OLLAMA_BASE_URL
    now = _time.monotonic()
    if cache_key in _model_cache:
        ts, cached = _model_cache[cache_key]
        if now - ts < _MODEL_CACHE_TTL:
            return cached
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                _model_cache[cache_key] = (now, models)
                return models
    except Exception:
        pass
    _model_cache[cache_key] = (now, [])
    return []


async def is_ollama_running() -> bool:
    """检查 Ollama API 是否可达。"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def list_ollama_model_details() -> list[dict]:
    """列出已下载的 Ollama 模型元数据。"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if r.status_code == 200:
                return r.json().get("models", [])
    except Exception:
        logger.warning("Ollama not available")
    return []


async def stream_ollama_model_pull(name: str):
    """以流式输出 Ollama 模型拉取进度事件。"""
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/pull", json={"name": name}) as response:
                if response.status_code != 200:
                    await response.aread()
                    yield {"error": f"Failed to pull model: HTTP {response.status_code}"}
                    return
                buffer = ""
                async for chunk in response.aiter_text():
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            total = data.get("total", 0)
                            completed = data.get("completed", 0)
                            progress = (completed / total * 100) if total > 0 else 0
                            yield {"status": data.get("status", ""), "progress": progress}
                        except Exception:
                            continue
    except Exception:
        yield {"error": "Failed to pull model"}


async def delete_ollama_model(name: str) -> tuple[bool, int, str]:
    """删除 Ollama 模型，返回 (成功, 状态码, 文本)。"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.request("DELETE", f"{OLLAMA_BASE_URL}/api/delete", json={"name": name})
            return r.status_code == 200, r.status_code, r.text
    except Exception:
        return False, 500, "Failed to delete model"


async def _resolve_ollama_model(target_model: str) -> Optional[str]:
    """解析 Ollama 模型名称，回退到最佳可用匹配。"""
    available = await get_ollama_models()
    if not available:
        return None
    if target_model in available:
        return target_model
    fallback = next(
        (m for m in available if target_model in m or m in target_model),
        available[0],
    )
    logger.debug("Ollama model '%s' not found, falling back to '%s'", target_model, fallback)
    return fallback


async def _generate_ollama(prompt: str, system: str = "", model: Optional[str] = None) -> str:
    """使用 Ollama /api/generate 端点生成文本。"""
    target_model = await _resolve_ollama_model(model or OLLAMA_MODEL)
    if not target_model:
        return ""

    payload = {
        "model": target_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
    }
    if system:
        payload["system"] = system

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            for attempt in range(3):
                r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
                if r.status_code != 200:
                    logger.warning("Ollama returned %d: %s", r.status_code, r.text[:200])
                    return ""
                try:
                    data = r.json() or {}
                except Exception:
                    data = {}
                text = (data.get("response") or "").strip()
                if text:
                    return text
                done_reason = str((data or {}).get("done_reason") or "").lower()
                if attempt < 2 and done_reason in {"load", "loading"}:
                    await asyncio.sleep(2.0)
                    continue
                break

            # 回退到 /api/chat
            chat_payload = {
                "model": target_model,
                "stream": False,
                "messages": (
                    ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}]
                ),
                "options": payload.get("options") or {},
            }
            r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=chat_payload)
            if r.status_code == 200:
                data = r.json() or {}
                return ((data.get("message") or {}).get("content") or "").strip()
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out after %ds", OLLAMA_TIMEOUT)
    except Exception:
        logger.warning("Ollama generation failed")
    return ""


async def _chat_ollama(messages: list[dict], system: str = "", model: Optional[str] = None) -> str:
    """使用 Ollama 原生消息数组进行对话。"""
    target_model = await _resolve_ollama_model(model or OLLAMA_MODEL)
    if not target_model:
        raise Exception(f"No Ollama models available at {OLLAMA_BASE_URL}")

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    chat_payload = {
        "model": target_model,
        "stream": False,
        "messages": payload_messages,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=chat_payload)
            if r.status_code == 200:
                data = r.json() or {}
                return ((data.get("message") or {}).get("content") or "").strip()
            else:
                raise Exception(f"Ollama returned {r.status_code}: {r.text[:200]}")
    except httpx.TimeoutException:
        raise Exception(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")


async def _stream_ollama(messages: list[dict], system: str = "", model: Optional[str] = None):
    """从 Ollama 逐块流式输出对话响应。"""
    target_model = await _resolve_ollama_model(model or OLLAMA_MODEL)
    if not target_model:
        yield "**SYSTEM ERROR:** No Ollama models available. Please start Ollama and pull a model."
        return

    payload_messages = []
    if system:
        payload_messages.append({"role": "system", "content": system})
    payload_messages.extend(messages)

    chat_payload = {
        "model": target_model,
        "stream": True,
        "messages": payload_messages,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 1024},
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=chat_payload) as r:
                if r.status_code != 200:
                    yield f"**SYSTEM ERROR:** Ollama returned {r.status_code}."
                    return
                async for line in r.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = (data.get("message") or {}).get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            pass
    except httpx.TimeoutException:
        yield "**SYSTEM TIMEOUT:** The LLM took too long to respond."
    except Exception:
        logger.warning("Ollama stream error")
        yield "**SYSTEM ERROR:** AI stream failed. Please try again later."


# ═════════════════════════════════════════════════════════════════
# TIER B: GEMINI（Google Cloud）
# ═════════════════════════════════════════════════════════════════

_gemini_configured = False
_gemini_model = None


def has_gemini_api_key() -> bool:
    """判断 Gemini 功能是否可用。"""
    key = GOOGLE_API_KEY.strip()
    if not key:
        return False
    if key in ("dummy", "your_gemini_api_key_here", "placeholder"):
        return False
    if key.startswith("your_"):
        return False
    return True


def has_deepseek_api_key() -> bool:
    """判断 DeepSeek API Key 是否已配置。"""
    key = DEEPSEEK_API_KEY.strip()
    return bool(key) and key not in ("dummy", "your_deepseek_api_key_here", "placeholder") and not key.startswith("your_")


def has_qwen_api_key() -> bool:
    """判断 Qwen / DashScope API Key 是否已配置。"""
    key = QWEN_API_KEY.strip()
    return bool(key) and key not in ("dummy", "your_qwen_api_key_here", "placeholder") and not key.startswith("your_")


def has_glm_api_key() -> bool:
    """判断智谱 GLM API Key 是否已配置。"""
    key = GLM_API_KEY.strip()
    return bool(key) and key not in ("dummy", "your_glm_api_key_here", "placeholder") and not key.startswith("your_")


def _get_gemini_model():
    """懒加载 Gemini 模型。"""
    global _gemini_configured, _gemini_model
    if _gemini_model:
        return _gemini_model
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "dummy":
        return None
    try:
        import google.generativeai as genai
        if not _gemini_configured:
            genai.configure(api_key=GOOGLE_API_KEY)
            _gemini_configured = True
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        return _gemini_model
    except Exception:
        logger.warning("Failed to initialize Gemini")
        return None


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """通过中心化 AI Provider 边界生成文本嵌入。

    这是一个同步函数，专门为无法异步的调用方（如启动期向量存储填充）设计。
    异步调用方应使用 ``asyncio.to_thread(embed_text, text)``。
    """
    global _gemini_configured
    if not has_gemini_api_key():
        # 尝试 DeepSeek / Qwen 嵌入，再回退到 Ollama
        return _openai_compatible_embed_fallback(text)

    import hashlib
    from .cache_service import cache as _cache

    try:
        _cache
    except Exception:
        _cache = None

    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    cache_key = f"emb:{task_type}:{text_hash}"

    if _cache:
        try:
            cached_val = _cache.get(cache_key)
            if cached_val is not None:
                return cached_val
        except Exception:
            pass

    try:
        import google.generativeai as genai
        if not _gemini_configured:
            genai.configure(api_key=GOOGLE_API_KEY)
            _gemini_configured = True
        result = genai.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        embedding = result.get("embedding") or [0.0] * 768
        if _cache:
            try:
                _cache.set(cache_key, embedding, ttl=86400)
            except Exception:
                pass
        return embedding
    except Exception:
        logger.error("Gemini embedding failed, trying fallbacks")
        return _openai_compatible_embed_fallback(text)


def _ollama_embed_fallback(text: str) -> list[float]:
    """使用 Ollama 生成嵌入（同步回退）。"""
    try:
        import requests
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json().get("embedding", [0.0] * 768)
    except Exception:
        pass
    # 零向量作为最后回退
    return [0.0] * 768


def _deepseek_embed(text: str) -> list[float]:
    """DeepSeek **不支持** Embedding API（纯对话模型，无 /embeddings 端点）。

    此函数始终返回零向量。请使用 Qwen DashScope（text-embedding-v3）
    或 Gemini（models/text-embedding-004）生成嵌入。
    """
    logger.debug("DeepSeek does not support embeddings — returning zero vector fallback")
    return [0.0] * 768


def _qwen_embed(text: str) -> list[float]:
    """使用 Qwen / DashScope API 生成嵌入（OpenAI 兼容 /embeddings 端点）。"""
    if not has_qwen_api_key():
        return [0.0] * 768
    try:
        import requests
        r = requests.post(
            f"{QWEN_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"},
            json={"model": QWEN_EMBEDDING_MODEL, "input": text},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("data", [{}])[0].get("embedding", [0.0] * 768)
        logger.debug("Qwen embed returned %d", r.status_code)
    except Exception:
        pass
    return [0.0] * 768


def _openai_compatible_embed_fallback(text: str) -> list[float]:
    """OpenAI 兼容嵌入回退链: Qwen → Ollama → 零向量。
    
    注意: DeepSeek 不支持嵌入 API，已从链中排除。
    """
    # Qwen DashScope (text-embedding-v3)
    emb = _qwen_embed(text)
    if any(v != 0.0 for v in emb[:3]):
        return emb
    # Ollama 本地
    return _ollama_embed_fallback(text)


async def embed_text_async(text: str, task_type: str = "retrieval_document") -> list[float]:
    """embed_text 的异步包装器。"""
    return await asyncio.to_thread(embed_text, text, task_type)


def generate_vision_content(prompt: str, image: Any, model: Optional[str] = None) -> str:
    """通过 Gemini Vision 从提示词+图像生成文本。"""
    if not has_gemini_api_key():
        return ""

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        vision_model = genai.GenerativeModel(model or GEMINI_VISION_MODEL)
        response = vision_model.generate_content([prompt, image])
        return (getattr(response, "text", "") or "").strip()
    except Exception:
        logger.error("Vision generation failed")
        return ""


async def generate_vision_content_async(prompt: str, image: Any, model: Optional[str] = None) -> str:
    """generate_vision_content 的异步包装器。"""
    return await asyncio.to_thread(generate_vision_content, prompt, image, model)


async def _generate_gemini(prompt: str, system: str = "") -> str:
    """使用 Google Gemini 生成文本。"""
    model = _get_gemini_model()
    if not model:
        return ""

    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    try:
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        return response.text.strip() if response.text else ""
    except Exception as e:
        err = str(e)
        if "429" in err or "Quota" in err:
            logger.warning("Gemini quota exceeded")
        else:
            logger.warning("Gemini generation failed")
        return ""


async def _chat_gemini(messages: list[dict], system: str = "") -> str:
    """使用 Gemini 原生多轮对话。"""
    model = _get_gemini_model()
    if not model:
        return ""

    try:
        from google.generativeai.types import ContentDict

        history: list[ContentDict] = []
        if system:
            history.append({"role": "user", "parts": [system]})
            history.append({"role": "model", "parts": ["Understood. I will follow those instructions."]})

        for msg in messages[:-1]:
            gemini_role = "model" if msg.get("role") == "assistant" else "user"
            history.append({"role": gemini_role, "parts": [msg.get("content", "")]})

        chat_session = await asyncio.to_thread(model.start_chat, history=history)
        last_msg = messages[-1].get("content", "") if messages else ""
        response = await asyncio.to_thread(chat_session.send_message, last_msg)
        return response.text.strip() if response.text else ""
    except Exception:
        logger.warning("Gemini chat failed")
        return ""


async def _stream_gemini(messages: list[dict], system: str = ""):
    """Gemini 伪流式（单块输出——Gemini SDK 不支持真正的 SSE）。"""
    result = await _chat_gemini(messages, system)
    if result:
        yield result


# ═════════════════════════════════════════════════════════════════
# TIER C: 云端 API（OpenAI / DeepSeek / Qwen / Anthropic / OpenRouter）
# ═════════════════════════════════════════════════════════════════

# OpenAI 兼容 API 的 base_url 和默认模型映射
_OPENAI_COMPATIBLE_PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "openrouter": ("https://openrouter.ai/api/v1", "google/gemini-2.5-flash"),
    "deepseek": (DEEPSEEK_BASE_URL, DEEPSEEK_MODEL),
    "qwen": (QWEN_BASE_URL, QWEN_MODEL),
    "dashscope": (QWEN_BASE_URL, QWEN_MODEL),
    "glm": (GLM_BASE_URL, GLM_MODEL),
    "bigmodel": (GLM_BASE_URL, GLM_MODEL),
}


async def _generate_cloud(prompt: str, system: str, model: Optional[str],
                          api_provider: str, api_key: str) -> str:
    """使用云服务商生成文本。支持 OpenAI / DeepSeek / Qwen / Anthropic / OpenRouter。"""
    try:
        provider_lower = api_provider.lower()
        async with httpx.AsyncClient(timeout=30) as client:
            # OpenAI 兼容协议（OpenAI, OpenRouter, DeepSeek, Qwen）
            if provider_lower in _OPENAI_COMPATIBLE_PROVIDERS:
                base_url, default_model = _OPENAI_COMPATIBLE_PROVIDERS[provider_lower]
                target_model = model or default_model
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload_messages = []
                if system:
                    payload_messages.append({"role": "system", "content": system})
                payload_messages.append({"role": "user", "content": prompt})

                r = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": target_model, "messages": payload_messages, "temperature": 0.7},
                )
                if r.status_code == 200:
                    data = r.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                logger.warning("%s error: %d — %s", api_provider, r.status_code, r.text[:200])

            elif api_provider.lower() == "anthropic":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                target_model = model or "claude-3-haiku-20240307"
                payload = {
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                    "temperature": 0.7,
                }
                if system:
                    payload["system"] = system

                r = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                if r.status_code == 200:
                    content = r.json().get("content", [])
                    if content:
                        return content[0].get("text", "").strip()
                logger.warning("Anthropic error: %d", r.status_code)

    except Exception:
        logger.warning("Cloud AI error (%s)", api_provider)
    return ""


async def _chat_cloud(messages: list[dict], system: str, model: Optional[str],
                      api_provider: str, api_key: str, max_retries: int = 3) -> str:
    """使用云服务商进行多轮对话。支持 OpenAI / DeepSeek / Qwen / Anthropic / OpenRouter。
    
    Args:
        max_retries: 最大重试次数，默认 3 次
    """
    provider_lower = api_provider.lower()
    last_error = None
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # OpenAI 兼容协议
                if provider_lower in _OPENAI_COMPATIBLE_PROVIDERS:
                    base_url, default_model = _OPENAI_COMPATIBLE_PROVIDERS[provider_lower]
                    target_model = model or default_model
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    payload_messages = []
                    if system:
                        payload_messages.append({"role": "system", "content": system})
                    payload_messages.extend(messages)

                    r = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={"model": target_model, "messages": payload_messages, "temperature": 0.7},
                    )
                    if r.status_code == 200:
                        data = r.json()
                        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    last_error = f"{api_provider} error: {r.status_code}"
                    logger.warning("Cloud AI attempt %d/%d failed: %s", attempt + 1, max_retries, last_error)

                elif api_provider.lower() == "anthropic":
                    headers = {
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }
                    target_model = model or "claude-3-haiku-20240307"
                    payload = {
                        "model": target_model,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                    }
                    if system:
                        payload["system"] = system

                    r = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                    if r.status_code == 200:
                        content = r.json().get("content", [])
                        if content:
                            return content[0].get("text", "").strip()
                    last_error = f"Anthropic error: {r.status_code}"
                    logger.warning("Cloud AI attempt %d/%d failed: %s", attempt + 1, max_retries, last_error)

            # 重试前等待（指数退避）
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
                
        except Exception as exc:
            last_error = str(exc)
            logger.warning("Cloud AI attempt %d/%d exception (%s): %s", attempt + 1, max_retries, api_provider, exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
    
    logger.warning("Cloud AI all %d retries exhausted for %s", max_retries, api_provider)
    return ""


async def _stream_cloud(messages: list[dict], system: str, model: Optional[str],
                        api_provider: str, api_key: str):
    """云端 API 单块输出。"""
    res = await _chat_cloud(messages, system, model, api_provider, api_key)
    if res:
        yield res


# ═════════════════════════════════════════════════════════════════
# 公共 API — 外部模块唯一允许调用的函数
# ═════════════════════════════════════════════════════════════════

async def is_available() -> bool:
    """检查是否有任何 AI 后端可用。"""
    ollama_models = await get_ollama_models()
    if ollama_models:
        return True
    if has_gemini_api_key():
        return True
    if has_deepseek_api_key():
        return True
    if has_qwen_api_key():
        return True
    if has_glm_api_key():
        return True
    return False


async def generate(
    prompt: str,
    system: str = "",
    model: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_guardrails: bool = True,
    enable_cache: bool = True,
) -> str:
    """
    使用最佳可用 AI 后端生成文本。

    降级链: 显式云提供商 → DeepSeek → Qwen → GLM → Ollama → Gemini → mock 响应。

    Args:
        prompt: 用户提示词
        system: 系统提示词
        model: 模型名称
        api_provider: 云提供商 (openai/deepseek/qwen/anthropic/openrouter)
        api_key: 云 API 密钥
        enable_guardrails: 是否启用安全护栏
        enable_cache: 是否启用语义缓存

    Returns:
        生成的文本
    """
    # 0. 安全护栏（可选）
    if enable_guardrails:
        try:
            from .guardrails import is_prompt_injection, redact_pii_from_text
            if is_prompt_injection(prompt):
                logger.warning("Prompt injection blocked")
                return "**SAFETY:** Input was flagged by security guardrails and blocked."
            redact_fn = redact_pii_from_text
        except ImportError:
            redact_fn = lambda x: x
    else:
        redact_fn = lambda x: x

    # 1. 语义缓存（可选）
    if enable_cache:
        try:
            from .semantic_cache import SemanticCache as _SC
            _cache = _SC()
            embedding = await asyncio.to_thread(embed_text, prompt)
            cached_response = _cache.lookup(prompt, embedding)
            if cached_response:
                return redact_fn(cached_response)
        except Exception:
            pass

    result = None

    # 显式云提供商覆盖
    if api_provider and api_key and api_provider.lower() not in ("ollama", "gemini"):
        result = await _generate_cloud(prompt, system, model, api_provider, api_key)

    # DeepSeek（从环境变量自动检测）
    if not result and has_deepseek_api_key():
        result = await _generate_cloud(prompt, system, model or DEEPSEEK_MODEL, "deepseek", DEEPSEEK_API_KEY)

    # Qwen（从环境变量自动检测）
    if not result and has_qwen_api_key():
        result = await _generate_cloud(prompt, system, model or QWEN_MODEL, "qwen", QWEN_API_KEY)

    # GLM 智谱（从环境变量自动检测）
    if not result and has_glm_api_key():
        result = await _generate_cloud(prompt, system, model or GLM_MODEL, "glm", GLM_API_KEY)

    # Tier A: Ollama
    if not result:
        ollama_models = await get_ollama_models()
        if ollama_models:
            result = await _generate_ollama(prompt, system, model)

    # Tier B: Gemini
    if not result:
        if has_gemini_api_key():
            result = await _generate_gemini(prompt, system)

    if result:
        result = redact_fn(result)
        # 保存到缓存
        if enable_cache:
            try:
                from .semantic_cache import SemanticCache as _SC
                _cache = _SC()
                embedding = await asyncio.to_thread(embed_text, prompt)
                _cache.add(prompt, embedding, result)
            except Exception:
                pass
        return result

    logger.warning("All AI backends unavailable for generate(), using mock fallback")
    mock_res = (
        "The system is running in offline mode. "
        "For full AI functionality, configure DEEPSEEK_API_KEY, QWEN_API_KEY, GOOGLE_API_KEY, "
        "or start Ollama."
    )
    return redact_fn(mock_res)


async def chat(
    messages: list[dict],
    system: str = "",
    model: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_guardrails: bool = True,
    enable_cache: bool = True,
) -> str:
    """
    使用最佳可用 AI 后端进行多轮对话。

    降级链: 显式云提供商 → DeepSeek → Qwen → GLM → Ollama → Gemini → mock 响应。
    """
    # 0. 安全护栏
    if enable_guardrails:
        try:
            from .guardrails import is_prompt_injection, redact_pii_from_text
            last_user_content = ""
            if messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user_content = msg.get("content", "")
                        break
                if not last_user_content:
                    last_user_content = messages[-1].get("content", "")
            if last_user_content and is_prompt_injection(last_user_content):
                logger.warning("Prompt injection blocked in chat")
                return "**SAFETY:** Input was flagged by security guardrails and blocked."
            redact_fn = redact_pii_from_text
        except ImportError:
            redact_fn = lambda x: x
    else:
        redact_fn = lambda x: x

    # 1. 语义缓存
    if enable_cache:
        try:
            from .semantic_cache import SemanticCache as _SC
            _cache = _SC()
            repr_text = messages[-1]["content"] if messages else ""
            embedding = await asyncio.to_thread(embed_text, repr_text)
            cached_response = _cache.lookup(json.dumps(messages), embedding)
            if cached_response:
                return redact_fn(cached_response)
        except Exception:
            pass

    result = None
    if api_provider and api_key and api_provider.lower() not in ("ollama", "gemini"):
        result = await _chat_cloud(messages, system, model, api_provider, api_key)

    # DeepSeek（从环境变量自动检测）
    if not result and has_deepseek_api_key():
        try:
            result = await _chat_cloud(messages, system, model or DEEPSEEK_MODEL, "deepseek", DEEPSEEK_API_KEY)
        except Exception:
            logger.warning("DeepSeek chat failed, trying next tier")

    # Qwen（从环境变量自动检测）
    if not result and has_qwen_api_key():
        try:
            result = await _chat_cloud(messages, system, model or QWEN_MODEL, "qwen", QWEN_API_KEY)
        except Exception:
            logger.warning("Qwen chat failed, trying next tier")

    # GLM 智谱（从环境变量自动检测）
    if not result and has_glm_api_key():
        try:
            result = await _chat_cloud(messages, system, model or GLM_MODEL, "glm", GLM_API_KEY)
        except Exception:
            logger.warning("GLM chat failed, trying next tier")

    if not result:
        ollama_models = await get_ollama_models()
        if ollama_models:
            try:
                result = await _chat_ollama(messages, system, model)
            except Exception:
                logger.warning("Ollama chat failed, trying Gemini")

    if not result:
        if has_gemini_api_key():
            result = await _chat_gemini(messages, system)

    if result:
        result = redact_fn(result)
        if enable_cache:
            try:
                from .semantic_cache import SemanticCache as _SC
                _cache = _SC()
                repr_text = messages[-1]["content"] if messages else ""
                embedding = await asyncio.to_thread(embed_text, repr_text)
                _cache.add(json.dumps(messages), embedding, result)
            except Exception:
                pass
        return result

    logger.warning("All AI backends unavailable for chat(), using mock response")
    return redact_fn("Hello! The system is running in offline mode. "
                     "Configure DEEPSEEK_API_KEY, QWEN_API_KEY, GOOGLE_API_KEY, "
                     "or start Ollama for full AI capabilities.")


async def chat_stream(
    messages: list[dict],
    system: str = "",
    model: Optional[str] = None,
    api_provider: Optional[str] = None,
    api_key: Optional[str] = None,
    enable_guardrails: bool = True,
) -> AsyncGenerator[str, None]:
    """
    流式多轮对话。逐块输出文本。

    降级: 显式云 → DeepSeek → Qwen → GLM → Ollama → Gemini → mock 流式响应。
    """
    # 0. 安全护栏
    if enable_guardrails:
        try:
            from .guardrails import is_prompt_injection, redact_pii_from_text
            last_user_content = ""
            if messages:
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        last_user_content = msg.get("content", "")
                        break
                if not last_user_content:
                    last_user_content = messages[-1].get("content", "")
            if last_user_content and is_prompt_injection(last_user_content):
                yield "**SAFETY:** Input was flagged by security guardrails and blocked."
                return
            _redact = redact_pii_from_text
        except ImportError:
            _redact = lambda x: x
    else:
        _redact = lambda x: x

    async def source_generator():
        if api_provider and api_key and api_provider.lower() not in ("ollama", "gemini"):
            async for chunk in _stream_cloud(messages, system, model, api_provider, api_key):
                yield chunk
            return

        # DeepSeek（从环境变量自动检测）
        if has_deepseek_api_key():
            has_yielded = False
            async for chunk in _stream_cloud(messages, system, model or DEEPSEEK_MODEL, "deepseek", DEEPSEEK_API_KEY):
                if chunk:
                    has_yielded = True
                    yield chunk
            if has_yielded:
                return

        # Qwen（从环境变量自动检测）
        if has_qwen_api_key():
            has_yielded = False
            async for chunk in _stream_cloud(messages, system, model or QWEN_MODEL, "qwen", QWEN_API_KEY):
                if chunk:
                    has_yielded = True
                    yield chunk
            if has_yielded:
                return

        # GLM 智谱（从环境变量自动检测）
        if has_glm_api_key():
            has_yielded = False
            async for chunk in _stream_cloud(messages, system, model or GLM_MODEL, "glm", GLM_API_KEY):
                if chunk:
                    has_yielded = True
                    yield chunk
            if has_yielded:
                return

        ollama_models = await get_ollama_models()
        if ollama_models:
            async for chunk in _stream_ollama(messages, system, model):
                yield chunk
            return

        if has_gemini_api_key():
            has_yielded = False
            async for chunk in _stream_gemini(messages, system):
                if chunk:
                    has_yielded = True
                    yield chunk
            if has_yielded:
                return

        yield ("Hello! The system is running in offline mode. "
               "Configure DEEPSEEK_API_KEY, QWEN_API_KEY, GOOGLE_API_KEY, "
               "or start Ollama for full AI capabilities.")

    # 带脱敏的流式包装
    async def redact_stream_generator(generator):
        buffer = ""
        async for chunk in generator:
            if not chunk:
                continue
            buffer += chunk
            redacted_buffer = _redact(buffer)
            split_idx = 0
            for i in range(len(redacted_buffer) - 1, -1, -1):
                char = redacted_buffer[i]
                if char in " \t\n\r.,;:!?()[]{}":
                    suffix = redacted_buffer[i + 1:]
                    if not any(c.isdigit() or c in "@-" for c in suffix):
                        split_idx = i + 1
                        break
            if split_idx > 0:
                yield redacted_buffer[:split_idx]
                buffer = redacted_buffer[split_idx:]
        if buffer:
            yield _redact(buffer)

    async for redacted_chunk in redact_stream_generator(source_generator()):
        yield redacted_chunk
