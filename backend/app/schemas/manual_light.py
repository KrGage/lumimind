"""Explicit, numeric targets for manual light control."""

import math
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ManualLightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: UUID
    cct_k: int = Field(strict=True, ge=1500, le=20000)
    illuminance_lux: float = Field(gt=0, le=30000, allow_inf_nan=False)
    session_id: str | None = Field(default=None, max_length=128)

    @field_validator("illuminance_lux", mode="before")
    @classmethod
    def require_numeric_lux(cls, value):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("illuminance_lux must be a finite JSON number")
        return value
