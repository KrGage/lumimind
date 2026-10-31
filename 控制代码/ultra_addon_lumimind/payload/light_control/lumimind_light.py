"""Single-lamp Fudan SDK adapter. Python 3.10+, standard library only.

calculate_recipe(lux, kelvin) calculates without networking.
set_light(lux, kelvin) calculates and sends one UDP datagram to the configured lamp.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
from pathlib import Path
import socket
import subprocess
from typing import Any
import uuid

_ROOT = Path(__file__).resolve().parent
_BRIDGE = _ROOT / "runtime" / "LightBridge.exe"
ConfigPath = str | os.PathLike[str] | None


class LightingError(RuntimeError):
    """Invalid configuration, failed recipe calculation, or failed UDP send."""


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LightingError(f"{name} must be a number, not text or boolean.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise LightingError(f"{name} must be finite.") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise LightingError(f"{name} must be finite and in {minimum}..{maximum}.")
    return number


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    number = _number(value, name, minimum, maximum)
    if not number.is_integer():
        raise LightingError(f"{name} must be an integer.")
    return int(number)


def _load_config(path: ConfigPath) -> dict[str, Any]:
    filename = Path(path) if path is not None else _ROOT / "light_config.json"
    try:
        config = json.loads(filename.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise LightingError(f"Cannot read configuration {filename}: {exc}") from exc
    if not isinstance(config, dict):
        raise LightingError("Configuration must be a JSON object.")
    return config


def _request(lux: float, cct: int, config: dict[str, Any]) -> dict[str, Any]:
    illuminance = _number(lux, "illuminance_lux", 0, float("inf"))
    if illuminance <= 0:
        raise LightingError("illuminance_lux must be greater than zero.")
    return {
        "illuminance_lux": illuminance,
        "cct_k": _integer(cct, "cct_k", 1500, 20000),
        "room_id": _integer(config.get("room_id"), "room_id", 1, 8),
        "min_ra": _number(config.get("min_ra", 80), "min_ra", 0, 100),
        "iteration_step": _integer(config.get("iteration_step", 50), "iteration_step", 1, 4095),
    }


def _calculate(request: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    timeout = _number(config.get("solve_timeout_seconds", 60), "solve_timeout_seconds", 0.1, 600)
    if os.name != "nt":
        raise LightingError("This SDK runtime requires Windows x64 with .NET Framework 4.7.2 or newer.")
    if not _BRIDGE.is_file():
        raise LightingError(f"Bridge runtime missing: {_BRIDGE}. Copy the complete package.")
    try:
        completed = subprocess.run(
            [str(_BRIDGE)], input=json.dumps(request, allow_nan=False),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(_BRIDGE.parent),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as exc:
        raise LightingError(f"Recipe calculation exceeded {timeout:g} seconds. No lamp command was sent.") from exc
    except OSError as exc:
        raise LightingError(f"Cannot start SDK bridge: {exc}") from exc
    try:
        result = json.loads(completed.stdout)
    except ValueError as exc:
        raise LightingError("Invalid SDK response. Check Windows/.NET and the complete runtime folder. "
                            + completed.stderr[-1200:]) from exc
    if not isinstance(result, dict):
        raise LightingError("SDK response must be a JSON object.")
    if completed.returncode != 0 or result.get("ok") is not True:
        raise LightingError(str(result.get("error", "SDK calculation failed.")))
    pwm = result.get("pwm")
    if (not isinstance(pwm, list) or len(pwm) != 5
            or any(type(v) is not int or not 0 <= v <= 4095 for v in pwm)):
        raise LightingError("SDK response contains invalid five-channel PWM values.")
    predicted = result.get("predicted")
    if not isinstance(predicted, dict):
        raise LightingError("SDK response is missing predicted optical metrics.")
    _number(predicted.get("illuminance_lux"), "predicted illuminance", 0, float("inf"))
    _number(predicted.get("cct_k"), "predicted color temperature", 0, float("inf"))
    result.update(status="calculated", hardware_confirmed=False)
    return result


def calculate_recipe(
    illuminance_lux: float, cct_k: int, *, config_path: ConfigPath = None,
) -> dict[str, Any]:
    """Compute a recipe without sending to the lamp. Raises LightingError on failure."""
    config = _load_config(config_path)
    return _calculate(_request(illuminance_lux, cct_k, config), config)


def set_light(
    illuminance_lux: float, cct_k: int, *, config_path: ConfigPath = None,
) -> dict[str, Any]:
    """Compute and send once. status='sent' means UDP sent, not measured light output.

    Each successful call sends one new command. Call once per final agent recommendation;
    never on each streamed token. SDK/validation errors send nothing and raise LightingError.
    """
    config = _load_config(config_path)
    request = _request(illuminance_lux, cct_k, config)
    raw_ip = config.get("lamp_ip")
    if not isinstance(raw_ip, str) or not raw_ip.strip():
        raise LightingError("Set lamp_ip in light_config.json to the actual single lamp IPv4 address.")
    try:
        address = ipaddress.IPv4Address(raw_ip)
    except ipaddress.AddressValueError as exc:
        raise LightingError("lamp_ip must be an IPv4 address, such as the lamp's configured LAN address.") from exc
    if address.is_multicast or address.is_unspecified or int(address) == 0xFFFFFFFF:
        raise LightingError("lamp_ip must be one unicast lamp address, not broadcast/multicast.")
    port = _integer(config.get("udp_port", 26666), "udp_port", 1, 65535)
    result = _calculate(request, config)
    message_id = uuid.uuid4().hex[:8]
    packet = {
        "Header": {"Namespace": "Opple", "Command": "PwmCtlRequest",
                   "MessageId": message_id, "Version": "1"},
        "Payload": {"RoomId": request["room_id"], "Channels": [
            {"Channel": i, "Value": value} for i, value in enumerate(result["pwm"])
        ]},
    }
    data = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
            udp.settimeout(3)
            sent = udp.sendto(data, (str(address), port))
        if sent != len(data):
            raise LightingError("Incomplete UDP send. Lamp execution is unconfirmed.")
    except OSError as exc:
        raise LightingError(f"UDP send failed; lamp execution is unconfirmed: {exc}") from exc
    result.update(status="sent", lamp_ip=str(address), udp_port=port, message_id=message_id)
    return result
