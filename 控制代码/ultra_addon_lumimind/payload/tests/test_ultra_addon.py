import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class AddonTests(unittest.TestCase):
    def test_agent_lifespan_and_chat_integration(self):
        from fastapi.testclient import TestClient
        from ultra_monitor.integration import load_config
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'backend'))
        original_cwd = Path.cwd()
        try:
            os.chdir(root / 'backend')
            with tempfile.TemporaryDirectory() as tmp:
                config = load_config()
                config.update(enabled=False, database=str(Path(tmp) / 'samples.db'))
                with patch('ultra_monitor.integration.load_config', return_value=config):
                    from app.main import app
                async def rag_chat(**kwargs):
                    self.assertIn('lumimind-light', kwargs['system'])
                    return {'reply': '设置方案。\n```lumimind-light\n{"cct_k":4000,"illuminance_lux":300}\n```'}
                fake_module = SimpleNamespace(rag_chat=rag_chat)
                with patch('app.main.init_db') as init, patch.dict(sys.modules, {'agent_framework.rag_pipeline': fake_module}), patch('ultra_monitor.lighting.set_light', return_value={'status': 'sent', 'hardware_confirmed': False}) as send:
                    with TestClient(app) as client:
                        init.assert_called_once()
                        self.assertEqual(client.get('/ultra/dashboard').status_code, 200)
                        self.assertEqual(client.get('/ultra/api/latest').status_code, 200)
                        response = client.post('/api/chat', json={'messages': [{'role': 'user', 'content': '调到4000K、300lx'}]})
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json()['lighting_target']['cct_k'], 4000)
                        self.assertEqual(response.json()['lighting']['status'], 'sent')
                        send.assert_called_once()
        finally:
            os.chdir(original_cwd)

    def test_installer_preview_backup_and_conflict(self):
        from install_ultra_addon import install, digest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, bundle = root / 'agent', root / 'bundle'
            destination = target / 'backend/app/main.py'
            source = bundle / 'payload/backend/app/main.py'
            destination.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            destination.write_text('old\n', encoding='utf-8')
            old_hash = digest(destination)
            source.write_text('new\n', encoding='utf-8')
            manifest = {'backend/app/main.py': {'new_hash': digest(source), 'old_hashes': [old_hash], 'required_existing': True}}
            (bundle / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            self.assertEqual(install(bundle, target), ['backend/app/main.py'])
            self.assertEqual(destination.read_text(), 'old\n')
            destination.write_text('custom\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                install(bundle, target, True)
            self.assertEqual(destination.read_text(), 'custom\n')
            destination.write_text('old\n', encoding='utf-8')
            install(bundle, target, True)
            self.assertEqual(destination.read_text(), 'new\n')
            self.assertEqual(next(target.glob('ultra_addon_backup_*/backend/app/main.py')).read_text(), 'old\n')
            self.assertEqual(install(bundle, target, True), [])

    def test_api_without_bluetooth_and_bad_limits(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from ultra_monitor.integration import load_config, register_ultra
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config.update(enabled=False, database='data/test.db')
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            app = FastAPI()
            register_ultra(app, config_path=path)
            register_ultra(app, config_path=path)
            with TestClient(app) as client:
                data = client.get('/ultra/api/latest').json()
                self.assertTrue(data['stale'])
                self.assertIsNone(data['sample'])
                self.assertEqual(data['connection']['state'], 'stopped')
                self.assertEqual(client.get('/ultra/api/history?limit=10001').status_code, 422)
                self.assertIn('ULTRA', client.get('/ultra/dashboard').text)
            self.assertIn('/ultra/api/latest', app.openapi()['paths'])

    def test_addon_exists(self):
        self.assertTrue((Path(__file__).resolve().parents[1] / 'ultra_monitor' / 'integration.py').is_file(),
                        '缺少可独立接入的 ultra 扩展')

    def test_decode_and_reject_bad_packets(self):
        from ultra_monitor.collector import decode, LUX_CCT
        self.assertEqual(decode(LUX_CCT, struct.pack('<fff', 321.5, 4000, 200)),
                         {'lux': 321.5, 'cct': 4000.0, 'eml': 200.0})
        self.assertEqual(decode(LUX_CCT, b'\x00'), {})
        self.assertEqual(decode(LUX_CCT, struct.pack('<fff', float('nan'), 0, -1)), {})

    def test_sqlite_history_and_restart(self):
        from ultra_monitor.storage import Store
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.db'
            store = Store(path)
            store.save({'device_id': 'ULTRA-004', 'measured_at': datetime.now(timezone.utc).isoformat(),
                        'triggered_at': '2026-10-26T23:59:48+00:00', 'lux': 321, 'cct': 4000})
            again = Store(path)
            self.assertEqual(again.latest()['lux'], 321)
            self.assertEqual(len(again.history(24 * 365 * 100, 10)), 1)
            with self.assertRaises(ValueError):
                store.history(1, 10001)

    def test_targets_strict_and_no_text_guessing(self):
        from ultra_monitor.lighting import split_target
        self.assertIsNone(split_target('建议 4000K，300 lux')[1])
        for payload in ({'cct_k': True, 'illuminance_lux': 300},
                        {'cct_k': 4000, 'illuminance_lux': -1},
                        {'cct_k': '4000', 'illuminance_lux': 300}):
            with self.assertRaises(ValueError):
                split_target('```lumimind-light\n' + json.dumps(payload) + '\n```')
        reply, target = split_target('设置为 4000K。\n```lumimind-light\n{"cct_k":4000,"illuminance_lux":300}\n```')
        self.assertEqual(reply, '设置为 4000K。')
        self.assertEqual(target['cct_k'], 4000)

    def test_truncated_second_target_or_trailing_comment_rejected(self):
        from ultra_monitor.lighting import split_target
        valid = '建议。\n```lumimind-light\n{"cct_k":4000,"illuminance_lux":300}\n```'
        for suffix in ('\n```lumimind-light\n{broken', '\n以上只是格式示例，不应执行。'):
            with self.assertRaises(ValueError):
                split_target(valid + suffix)

    def test_one_send_and_failure_preserves_reply(self):
        from ultra_monitor.lighting import LightingService
        async def run():
            service = LightingService()
            text = '建议已生成。\n```lumimind-light\n{"cct_k":4000,"illuminance_lux":300}\n```'
            with patch('ultra_monitor.lighting.set_light', return_value={'status': 'sent', 'hardware_confirmed': False}) as send:
                result = await service.process(text)
                self.assertEqual(send.call_count, 1)
                self.assertEqual(result['lighting']['status'], 'sent')
            with patch('ultra_monitor.lighting.set_light', side_effect=RuntimeError('未配置 IP')):
                result = await service.process(text)
                self.assertEqual(result['reply'], '建议已生成。')
                self.assertEqual(result['lighting']['status'], 'error')
        asyncio.run(run())

    def test_invalid_target_never_sends(self):
        from ultra_monitor.lighting import LightingService
        async def run():
            service = LightingService()
            with patch('ultra_monitor.lighting.set_light') as send:
                result = await service.process('```lumimind-light\n{"cct_k":true,"illuminance_lux":300}\n```')
                self.assertEqual(result['lighting']['status'], 'invalid_target')
                await service.process('普通知识问答，不需要调光。')
                send.assert_not_called()
        asyncio.run(run())

    def test_collector_optical_sample_and_optional_notify(self):
        from ultra_monitor.collector import Collector, LUX_CCT, SPD_TRIGGER, FLK_TRIGGER
        from ultra_monitor.integration import load_config
        from ultra_monitor.storage import Store
        import bleak
        class FakeClient:
            def __init__(self, *args, **kwargs):
                self.is_connected = True
                self.handlers = {}
                self.services = [SimpleNamespace(characteristics=[
                    SimpleNamespace(uuid=LUX_CCT, properties=['notify']),
                    SimpleNamespace(uuid='unused', properties=['notify'])])]
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                self.is_connected = False
            async def start_notify(self, char, handler):
                self.handlers[char.uuid] = handler
            async def write_gatt_char(self, uuid, data):
                if uuid == SPD_TRIGGER:
                    self.handlers[LUX_CCT](None, bytearray(struct.pack('<fff', 333, 4100, 100)))
                elif uuid == FLK_TRIGGER:
                    raise RuntimeError('optional trigger unavailable')
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                config = load_config()
                store = Store(Path(tmp) / 'samples.db')
                collector = Collector(config, store)
                async def scan(*args, **kwargs):
                    return object()
                async def sleep(seconds):
                    if store.latest() is not None:
                        raise asyncio.CancelledError()
                with patch.object(bleak, 'BleakClient', FakeClient), patch.object(bleak.BleakScanner, 'find_device_by_address', scan), patch('ultra_monitor.collector.asyncio.sleep', sleep):
                    with self.assertRaises(asyncio.CancelledError):
                        await collector.collect()
                sample = store.latest()
                self.assertEqual(sample['lux'], 333)
                self.assertEqual(sample['cct'], 4100)
                self.assertLessEqual(sample['triggered_at'], sample['measured_at'])
        asyncio.run(run())

    def test_reconnect_and_shutdown(self):
        from ultra_monitor.collector import Collector
        from ultra_monitor.integration import load_config
        from ultra_monitor.storage import Store
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                collector = Collector(load_config(), Store(Path(tmp) / 'samples.db'))
                attempts = []
                async def collect():
                    attempts.append(1)
                    if len(attempts) == 1:
                        raise RuntimeError('disconnected')
                    raise asyncio.CancelledError()
                async def sleep(seconds):
                    pass
                with patch.object(collector, 'collect', collect), patch('ultra_monitor.collector.asyncio.sleep', sleep):
                    with self.assertRaises(asyncio.CancelledError):
                        await collector.run()
                self.assertEqual(len(attempts), 2)
                collector.task = asyncio.create_task(asyncio.sleep(60))
                await collector.stop()
                self.assertIsNone(collector.task)
                self.assertEqual(collector.status['state'], 'stopped')
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
