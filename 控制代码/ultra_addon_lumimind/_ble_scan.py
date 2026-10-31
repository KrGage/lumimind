"""List nearby BLE devices to locate the ULTRA sensor."""
import asyncio

from bleak import BleakScanner


async def main() -> None:
    print('扫描周围 BLE 设备（10 秒）...')
    devices = await BleakScanner.discover(timeout=10)
    if not devices:
        print('未发现任何 BLE 设备：请确认 Windows 蓝牙已开启。')
        return
    for d in devices:
        print(f'{d.address}  {d.name!r}')


asyncio.run(main())
