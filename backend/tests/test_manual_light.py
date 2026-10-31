"""Manual API tests use a temporary database and never contact a lamp."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.manual_light_api import apply_manual_light, latest_manual_light, router  # noqa: E402
from app.db.database import Base  # noqa: E402
from app.schemas.manual_light import ManualLightRequest  # noqa: E402


class ManualLightApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.request = Request({
            "type": "http", "method": "POST", "scheme": "http",
            "path": "/api/light/manual", "query_string": b"", "headers": [],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
        })

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def payload(self):
        return {
            "command_id": str(uuid4()),
            "cct_k": 4000,
            "illuminance_lux": 300,
            "session_id": None,
        }

    def test_sends_once_and_replay_returns_same_record(self):
        body = self.payload()
        with patch("app.api.manual_light_api.set_light", return_value={
            "status": "sent", "hardware_confirmed": False, "pwm": [1, 2, 3, 4, 5]
        }) as send:
            request = ManualLightRequest.model_validate(body)
            first = apply_manual_light(request, self.request, self.db)
            second = apply_manual_light(request, self.request, self.db)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "sent")
        self.assertFalse(first["hardware_confirmed"])
        self.assertEqual(latest_manual_light(self.db)["command_id"], body["command_id"])
        send.assert_called_once_with(300.0, 4000)

    def test_invalid_target_never_sends(self):
        with patch("app.api.manual_light_api.set_light") as send:
            for field, invalid in (("cct_k", 4000.5), ("illuminance_lux", "300"),
                                   ("illuminance_lux", True), ("illuminance_lux", 0)):
                body = self.payload()
                body[field] = invalid
                with self.assertRaises(ValidationError):
                    ManualLightRequest.model_validate(body)
        send.assert_not_called()

    def test_failure_is_recorded_and_not_replayed(self):
        from light_control.lumimind_light import LightingDeliveryUnknown, LightingError

        for exception, expected_status in ((LightingError("SDK failed"), "failed"),
                                           (LightingDeliveryUnknown("UDP uncertain"), "unknown")):
            body = self.payload()
            with patch("app.api.manual_light_api.set_light", side_effect=exception) as send:
                request = ManualLightRequest.model_validate(body)
                with self.assertRaises(HTTPException) as first:
                    apply_manual_light(request, self.request, self.db)
                with self.assertRaises(HTTPException) as second:
                    apply_manual_light(request, self.request, self.db)
            self.assertEqual(first.exception.status_code, 503)
            self.assertEqual(first.exception.detail["status"], expected_status)
            self.assertEqual(second.exception.status_code, 409)
            send.assert_called_once()

    def test_route_is_registered(self):
        self.assertIn("/api/light/manual", [route.path for route in router.routes])

    def test_remote_caller_cannot_send(self):
        remote = Request({
            "type": "http", "method": "POST", "scheme": "http",
            "path": "/api/light/manual", "query_string": b"", "headers": [],
            "client": ("192.0.2.10", 12345), "server": ("127.0.0.1", 8000),
        })
        with patch("app.api.manual_light_api.set_light") as send:
            with self.assertRaises(HTTPException) as blocked:
                apply_manual_light(ManualLightRequest.model_validate(self.payload()), remote, self.db)
        self.assertEqual(blocked.exception.status_code, 403)
        send.assert_not_called()

    def test_other_website_cannot_trigger_local_lamp(self):
        outside_origin = Request({
            "type": "http", "method": "POST", "scheme": "http",
            "path": "/api/light/manual", "query_string": b"",
            "headers": [(b"origin", b"https://example.org")],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
        })
        with patch("app.api.manual_light_api.set_light") as send:
            with self.assertRaises(HTTPException) as blocked:
                apply_manual_light(ManualLightRequest.model_validate(self.payload()), outside_origin, self.db)
        self.assertEqual(blocked.exception.status_code, 403)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
