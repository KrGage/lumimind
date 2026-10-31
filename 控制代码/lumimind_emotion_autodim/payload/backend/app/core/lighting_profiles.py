"""Numeric targets for the existing two-mode emotion demonstration agent.

An upstream agent's explicit target takes precedence. Profiles are configurable
demonstration values, not an optimization algorithm or a clinical prescription.
"""

import json
from pathlib import Path

from app.schemas.agent import AgentAction, LightMode
from app.schemas.lighting import LightingTarget

TARGET_MODES = {
    AgentAction.switch_to_adjustment_light: LightMode.adjustment_light,
    AgentAction.switch_to_normal_office_light: LightMode.normal_office_light,
}
_CONFIG = Path(__file__).resolve().parents[3] / "light_control" / "light_config.json"
_DEFAULTS = {
    "normal_office_light": {"cct_k": 4500, "illuminance_lux": 300},
    "adjustment_light": {"cct_k": 5000, "illuminance_lux": 300},
}


def target_for_action(action: AgentAction) -> LightingTarget | None:
    mode = TARGET_MODES.get(action)
    if mode is None:
        return None
    config = json.loads(_CONFIG.read_text(encoding="utf-8-sig"))
    profiles = config.get("emotion_profiles", _DEFAULTS)
    return LightingTarget.model_validate(profiles[mode.value])
