"""ULTRA protocol adapted from well/collector/ble_collector.py (read-only source).

All notify subscriptions are retained: LGT-01 may require subscription to otherwise
unused notify characteristics before responding. Measurements are NOT control feedback.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import math
import struct
from typing import Any

from .storage import Store

log = logging.getLogger(__name__)
SUFFIX = '-dfd7-4c73-b97d-fd9934c2da51'
SPD_TRIGGER = 'b0531401' + SUFFIX
FLK_TRIGGER = 'b0531430' + SUFFIX
LUX_CCT = 'b0531409' + SUFFIX
FIELDS = {
    LUX_CCT: ('lux', 'cct', 'eml'),
    'b053140b' + SUFFIX: ('rf', 'rg'),
    'b0531410' + SUFFIX: ('caf',),
    'b0531413' + SUFFIX: ('cs', 'm_edi'),
    'b0531432' + SUFFIX: ('frequency',),
    'b0531433' + SUFFIX: ('flicker',),
    'b0531434' + SUFFIX: ('percent_flicker',),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode(uuid: str, data: bytes) -> dict[str, float]:
    names = FIELDS.get(uuid.lower(), ())
    if not names or len(data) < len(names) * 4:
        return {}
    values = struct.unpack_from('<' + 'f' * len(names), data)
    result = {name: float(value) for name, value in zip(names, values)
              if math.isfinite(value) and value >= 0 and (name != 'cct' or value > 0)}
    # CCT and illuminance must come from one complete, valid optical packet.
    if uuid.lower() == LUX_CCT and not {'lux', 'cct'} <= result.keys():
        return {}
    return result


class Collector:
    def __init__(self, config: dict[str, Any], store: Store):
        self.config = config
        self.store = store
        self.status: dict[str, Any] = {'state': 'stopped', 'error': None, 'last_received_at': None,
                                       'device_id': config['device_id'], 'ble_address': config['ble_address']}
        self.task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.config['enabled'] and (self.task is None or self.task.done()):
            self.task = asyncio.create_task(self.run(), name='ultra-single-device')

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        self.status['state'] = 'stopped'

    async def run(self) -> None:
        while True:
            try:
                await self.collect()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.status.update(state='disconnected', error=str(exc))
                log.warning('ULTRA disconnected; retrying: %s', exc)
            await asyncio.sleep(self.config['retry_seconds'])

    async def collect(self) -> None:
        from bleak import BleakClient, BleakScanner

        self.status.update(state='connecting', error=None)
        device = await BleakScanner.find_device_by_address(self.config['ble_address'], timeout=15)
        if device is None:
            raise RuntimeError('未发现 ultra，请确认供电、蓝牙和设备地址，关闭占用设备的其他采集程序。')
        async with BleakClient(device, timeout=20) as client:
            batch: dict[str, float] = {}
            received: str | None = None
            accepting = False

            def notify(uuid: str):
                def handler(characteristic: Any, data: bytearray) -> None:
                    nonlocal received
                    if not accepting:
                        return
                    parsed = decode(uuid, bytes(data))
                    if parsed:
                        batch.update(parsed)
                        if uuid == LUX_CCT and 'lux' in parsed and 'cct' in parsed:
                            received = now()
                return handler

            subscribed: set[str] = set()
            for service in client.services:
                for characteristic in service.characteristics:
                    if 'notify' in characteristic.properties:
                        uuid = characteristic.uuid.lower()
                        try:
                            await client.start_notify(characteristic, notify(uuid))
                            subscribed.add(uuid)
                        except Exception as exc:
                            log.warning('ULTRA subscribe %s: %s', uuid, exc)
            if LUX_CCT not in subscribed:
                raise RuntimeError('ultra 关键照度/CCT 特征订阅失败。')
            while client.is_connected:
                batch.clear()
                received = None
                triggered_at = now()
                self.status.update(state='measuring', error=None)
                accepting = True
                await client.write_gatt_char(SPD_TRIGGER, b'\x01')
                # Frequency is optional; do not discard optical data on unsupported flicker trigger.
                try:
                    await client.write_gatt_char(FLK_TRIGGER, b'\x01')
                except Exception as exc:
                    log.warning('ULTRA flicker trigger: %s', exc)
                await asyncio.sleep(self.config['interval_seconds'])
                accepting = False
                if not client.is_connected:
                    raise RuntimeError('ultra 蓝牙连接已断开。')
                if received is None:
                    raise RuntimeError('本轮未收到有效照度/CCT 数据，将重新连接。')
                sample = {'device_id': self.config['device_id'], 'triggered_at': triggered_at,
                          'measured_at': received, **batch}
                await asyncio.to_thread(self.store.save, sample)
                self.status.update(state='measuring', error=None, last_received_at=received)
            raise RuntimeError('ultra 蓝牙连接已断开。')
