#!/usr/bin/env python3
"""
ppt-master 图像生成后端测试脚本

测试 GLM (智谱) 和 OpenAI (GPT-Image) 两个后端的连通性和生成能力。
配置位置（按优先级自动合并）：
  1. 项目根目录 .env           （项目级覆盖）
  2. .qoder/skills/ppt-master/.env  （skill 级共享，推荐）
  3. ~/.ppt-master/.env       （用户级全局默认）

请在上述任一位置填入真实的 API Key：
  - ZHIPU_API_KEY=xxx
  - OPENAI_API_KEY=xxx

用法:
  python test_image_gen.py              # 测试所有已配置的后端
  python test_image_gen.py zhipu        # 仅测试 GLM
  python test_image_gen.py openai       # 仅测试 OpenAI
  python test_image_gen.py openai --prompt "赛博朋克城市" --size 16:9
"""
from __future__ import annotations

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# ── 路径常量 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_DIR = Path.home() / ".qoder" / "skills" / "ppt-master"
SCRIPTS_DIR = SKILL_DIR / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "projects" / "image_test_output"

PYTHON = r"D:\ProgramData\conda_env\python39\python.exe"

# ── 后端配置 ──────────────────────────────────────────────
BACKENDS = {
    "zhipu": {
        "label": "智谱 GLM-Image",
        "env_key": "ZHIPU_API_KEY",
        "default_model": "glm-image",
        "placeholder": "your_zhipu_api_key_here",
        "key_url": "https://open.bigmodel.cn/usercenter/apikeys",
    },
    "openai": {
        "label": "OpenAI GPT-Image",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-image-2",
        "placeholder": "your_openai_api_key_here",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "qwen": {
        "label": "通义千问 Qwen Image",
        "env_key": "QWEN_API_KEY",
        "default_model": "qwen-image-2.0-pro",
        "placeholder": "your_qwen_api_key_here",
        "key_url": "https://dashscope.console.aliyun.com/apiKey",
    },
}

# ── 测试用 Prompt（中英文对照）────────────────────────────
DEFAULT_PROMPT_ZH = "科技感的AI神经网络节点连接图，蓝色和紫色渐变光效，深色背景，未来主义风格"
DEFAULT_PROMPT_EN = "Futuristic AI neural network with glowing nodes, blue and purple gradient, dark background, cyberpunk style"


def load_env() -> dict[str, str]:
    """从所有候选 .env 合并加载环境变量（高优先级覆盖低优先级）。

    查找顺序（从高到低）：
      1. 项目根目录 .env
      2. .qoder/skills/ppt-master/.env  ← skill 级共享配置
      3. ~/.ppt-master/.env
    """
    candidates = [
        PROJECT_ROOT / ".env",
        SKILL_DIR / ".env",                    # 用户级 skill 的 .env
        Path.home() / ".ppt-master" / ".env",
    ]

    loaded = {}
    # 反向遍历：低优先级先加载，高优先级覆盖
    for env_path in reversed(candidates):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 去掉行内注释
            if " #" in value:
                value = value.split(" #")[0].strip()
            loaded[key] = value
            if key not in os.environ:
                os.environ[key] = value

    if not loaded:
        print("[ERROR] 未找到任何 .env 文件，已查找：")
        for c in candidates:
            print(f"  - {c} ({'存在' if c.exists() else '不存在'})")
        sys.exit(1)

    return loaded


def check_backend_config(backend: str, env_vars: dict) -> bool:
    """检查指定后端的 API Key 是否已正确配置。"""
    cfg = BACKENDS[backend]
    api_key = env_vars.get(cfg["env_key"], "")

    if not api_key or api_key == cfg["placeholder"]:
        print(f"\n  [{backend}] {cfg['label']}")
        print(f"  状态: ❌ 未配置（占位符值）")
        print(f"  获取 Key: {cfg['key_url']}")
        print(f"  请在 .env 中设置 {cfg['env_key']}=<你的真实Key>")
        return False

    # 脱敏显示 Key 前后几位
    masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"\n  [{backend}] {cfg['label']}")
    print(f"  状态: ✅ 已配置")
    print(f"  Key: {masked}")
    print(f"  默认模型: {cfg['default_model']}")
    return True


def run_image_gen(
    backend: str,   
    prompt: str,
    aspect_ratio: str = "16:9",
    image_size: str = "1K",
) -> dict:
    """调用 ppt-master 的 image_gen.py 生成图片。

    Returns:
        {"success": bool, "elapsed": float, "output_file": str | None, "error": str | None}
    """
    result = {"success": False, "elapsed": 0.0, "output_file": None, "error": None}

    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_gen_script = SCRIPTS_DIR / "image_gen.py"
    if not image_gen_script.exists():
        result["error"] = f"image_gen.py 不存在: {image_gen_script}"
        return result

    # 构建命令
    cmd = [
        PYTHON,
        str(image_gen_script),
        prompt,
        "--backend", backend,
        "--aspect_ratio", aspect_ratio,
        "--image_size", image_size,
        "-o", str(OUTPUT_DIR),
    ]

    print(f"  命令: {' '.join(cmd[:6])}...")
    print(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
            encoding="utf-8",
            errors="replace",
        )
        result["elapsed"] = time.time() - start

        if proc.returncode == 0:
            result["success"] = True
            # 从 stdout 中提取输出文件路径
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    result["output_file"] = line
                    break
                if "saved" in line.lower() or "output" in line.lower():
                    # 尝试提取路径
                    for token in line.split():
                        if any(token.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                            result["output_file"] = token
                            break

            print(f"  结果: ✅ 生成成功（{result['elapsed']:.1f}s）")
            if result["output_file"]:
                print(f"  文件: {result['output_file']}")
            # 打印前几行输出
            for line in proc.stdout.strip().splitlines()[:5]:
                print(f"  > {line}")
        else:
            result["error"] = proc.stderr.strip() or proc.stdout.strip()
            print(f"  结果: ❌ 生成失败（{result['elapsed']:.1f}s）")
            # 打印错误信息（截取关键部分）
            error_lines = (proc.stderr + "\n" + proc.stdout).strip().splitlines()
            for line in error_lines[-10:]:
                if line.strip():
                    print(f"  ! {line.strip()}")

    except subprocess.TimeoutExpired:
        result["elapsed"] = time.time() - start
        result["error"] = "超时（120s）"
        print(f"  结果: ❌ 请求超时（{result['elapsed']:.1f}s）")
    except Exception as e:
        result["elapsed"] = time.time() - start
        result["error"] = str(e)
        print(f"  结果: ❌ 异常: {e}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ppt-master 图像生成后端测试")
    parser.add_argument(
        "backend",
        nargs="?",
        default="all",
        choices=["all", "zhipu", "openai", "qwen"],
        help="测试的后端（默认 all）",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="自定义生成提示词",
    )
    parser.add_argument(
        "--ratio",
        default="16:9",
        help="宽高比（默认 16:9）",
    )
    parser.add_argument(
        "--size",
        default="1K",
        help="图片尺寸档位（512px | 1K | 2K | 4K）",
    )
    args = parser.parse_args()

    # ── 打印头部 ──────────────────────────────────
    print("=" * 64)
    print("  ppt-master 图像生成后端测试")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {PYTHON}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 64)

    # ── 加载 .env ──────────────────────────────────
    env_vars = load_env()

    # ── 决定测试哪些后端 ────────────────────────────
    if args.backend == "all":
        test_backends = list(BACKENDS.keys())
    else:
        test_backends = [args.backend]

    # ── 选择 prompt ────────────────────────────────
    prompt = args.prompt or DEFAULT_PROMPT_ZH

    # ── 逐个检查 + 测试 ────────────────────────────
    results = {}
    ready_backends = []

    for backend in test_backends:
        print(f"\n{'─' * 64}")
        print(f"  检查后端: {BACKENDS[backend]['label']}")

        # 检查配置
        if not check_backend_config(backend, env_vars):
            results[backend] = {"success": False, "skipped": True}
            continue

        ready_backends.append(backend)

    # ── 对已配置的后端执行生成测试 ──────────────────
    for backend in ready_backends:
        print(f"\n{'─' * 64}")
        print(f"  生成测试: {BACKENDS[backend]['label']}")
        print(f"  尺寸: {args.ratio} @ {args.size}")
        results[backend] = run_image_gen(
            backend, prompt, aspect_ratio=args.ratio, image_size=args.size
        )

    # ── 汇总报告 ──────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  测试汇总")
    print(f"{'=' * 64}")

    total = len(results)
    passed = sum(1 for r in results.values() if r.get("success"))
    skipped = sum(1 for r in results.values() if r.get("skipped"))
    failed = total - passed - skipped

    for backend, r in results.items():
        label = BACKENDS[backend]["label"]
        if r.get("skipped"):
            status = "⏭ 跳过（未配置 Key）"
        elif r.get("success"):
            elapsed = r.get("elapsed", 0)
            status = f"✅ 通过（{elapsed:.1f}s）"
        else:
            status = "❌ 失败"
        print(f"  {label:30s} {status}")

    print(f"\n  总计: {total} | 通过: {passed} | 跳过: {skipped} | 失败: {failed}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 64)

    # 退出码
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
