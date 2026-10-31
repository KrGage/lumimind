"""BLE connectivity test for the ULTRA sensor. Runs one real measurement round."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'payload'))

from ultra_monitor.collector import Collector, LUX_CCT, SPD_TRIGGER, FLK_TRIGGER, decode  # noqa: E402


async def main() -> None:
    from bleak import BleakClient, BleakScanner

    address = 'F6:2B:AC:6B:FC:66'
    print(f'[1/4] 扫描 BLE 设备 {address} ...（最长 15 秒）')
    device = await BleakScanner.find_device_by_address(address, timeout=15)
    if device is None:
        print('未发现该地址的设备。请确认：传感器已供电、Windows 蓝牙已开启、地址是否匹配。')
        return
    print(f'发现设备: {device.name or "(未广播名称)"}')

    print('[2/4] 建立连接 ...')
    async with BleakClient(device, timeout=20) as client:
        print(f'已连接: {client.is_connected}')
        batch: dict[str, float] = {}
        received = False
        accepting = False

        def notify(uuid: str):
            def handler(_char, data: bytearray) -> None:
                nonlocal received
                if not accepting:
                    return
                parsed = decode(uuid, bytes(data))
                if parsed:
                    batch.update(parsed)
                    if uuid == LUX_CCT and 'lux' in parsed and 'cct' in parsed:
                        received = True
            return handler

        print('[3/4] 订阅 notify 特征 ...')
        subscribed = []
        for service in client.services:
            for characteristic in service.characteristics:
                if 'notify' in characteristic.properties:
                    uuid = characteristic.uuid.lower()
                    try:
                        await client.start_notify(characteristic, notify(uuid))
                        subscribed.append(uuid)
                    except Exception as exc:
                        print(f'  订阅失败 {uuid}: {exc}')
        print(f'已订阅 {len(subscribed)} 个特征')
        if LUX_CCT not in subscribed:
            print('关键照度/CCT 特征订阅失败，中止。')
            return

        print('[4/4] 触发一轮测量（12 秒采集窗口）...')
        accepting = True
        await client.write_gatt_char(SPD_TRIGGER, b'\x01')
        try:
            await client.write_gatt_char(FLK_TRIGGER, b'\x01')
        except Exception as exc:
            print(f'  频闪触发不支持（可忽略）: {exc}')
        await asyncio.sleep(12)
        accepting = False

    if received:
        print('测量成功，本轮数据:')
        print(json.dumps(batch, ensure_ascii=False, indent=2))
    else:
        print('已连接但本轮未收到有效照度/CCT 数据。')


if __name__ == '__main__':
    asyncio.run(main())
