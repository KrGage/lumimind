"""快速测试 DeepSeek API 是否可用"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ai_agent"))

import asyncio
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "ai_agent", ".env"))

async def test():
    import httpx

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    print(f"API Key: {api_key[:12]}...")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是 LumiMind 医学健康智能助手。始终使用中文回答。"},
            {"role": "user", "content": "帮我分析一下我之前的生理数据"},
        ],
        "max_tokens": 500,
    }

    print(f"\n正在调用 DeepSeek API...")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
            print(f"\n✅ DeepSeek 回复:\n{reply}")
        else:
            print(f"❌ 错误: {resp.text[:500]}")

asyncio.run(test())
