"""Strict machine-readable targets; reuse the existing SDK/UDP implementation."""
from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

from light_control.lumimind_light import set_light

CONTROL_INSTRUCTION = '''
光控集成规则：当本轮用户要求实际调光，或明确请求为当前环境生成可执行的光照方案时，
选定一组具体的目标照度和色温，在中文回复末尾附加且只附加一个机器数据块：
```lumimind-light
{"cct_k":4000,"illuminance_lux":300}
```
数值必须是确定的 JSON 数字，cct_k 为 1500..20000 的整数，illuminance_lux 大于零且至多 30000。
纯知识问答、询问历史、解释已有方案、引用他人命令、信息不足时不要附加此块。
此块会自动控制真实灯具。回复描述目标方案，不声称已发送、实测达标或已改善情绪。
不要根据传感器实测进行自动纠偏。实际下发状态由程序返回。
'''
BLOCK = re.compile(r'```lumimind-light\s*\n(.*?)\n```', re.DOTALL)


def split_target(text: str) -> tuple[str, dict[str, Any] | None]:
    matches = list(BLOCK.finditer(text))
    if not matches:
        if '```lumimind-light' in text:
            raise ValueError('调光目标数据块不完整，未发送。')
        return text, None
    if len(matches) != 1 or text.count('```lumimind-light') != 1:
        raise ValueError('存在多组或截断的调光目标，未发送。')
    if text[matches[0].end():].strip():
        raise ValueError('调光目标必须是回复末尾的唯一执行数据块。')
    try:
        target = json.loads(matches[0].group(1))
    except ValueError as exc:
        raise ValueError('调光目标 JSON 无效，未发送。') from exc
    if not isinstance(target, dict) or set(target) != {'cct_k', 'illuminance_lux'}:
        raise ValueError('调光目标必须包含 cct_k 和 illuminance_lux。')
    cct, lux = target['cct_k'], target['illuminance_lux']
    if type(cct) is not int or not 1500 <= cct <= 20000:
        raise ValueError('目标 CCT 必须是 1500..20000 的整数。')
    if type(lux) not in (float, int) or not 0 < lux <= 30000 or not math.isfinite(lux):
        raise ValueError('目标照度必须是大于 0、至多 30000 的有限数字。')
    return BLOCK.sub('', text).strip(), target


class LightingService:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.last: dict[str, Any] | None = None

    async def process(self, text: str) -> dict[str, Any]:
        try:
            reply, target = split_target(text)
        except ValueError as exc:
            return {'reply': BLOCK.sub('', text).strip(), 'lighting_target': None,
                    'lighting': {'status': 'invalid_target', 'error': str(exc), 'hardware_confirmed': False}}
        if target is None:
            return {'reply': reply, 'lighting_target': None, 'lighting': None}
        async with self.lock:
            try:
                result = await asyncio.to_thread(set_light, target['illuminance_lux'], target['cct_k'])
            except Exception as exc:
                result = {'status': 'error', 'error': str(exc), 'hardware_confirmed': False}
            self.last = {'target': target, 'result': result}
        label = '控制报文已发送；实际出光请查看 dashboard。' if result.get('status') == 'sent' else '未完成灯控下发：' + result.get('error', '未知错误')
        # Preserve reply text on errors; callers/frontends receive a separate status too.
        return {'reply': reply, 'lighting_target': target, 'lighting': {**result, 'message': label}}


lighting_service = LightingService()
