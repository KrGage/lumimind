from pydantic import BaseModel, ConfigDict, Field


class LightingTarget(BaseModel):
    cct_k: int = Field(strict=True, ge=1500, le=20000)
    illuminance_lux: float = Field(strict=True, gt=0, allow_inf_nan=False)


class AcceptLighting(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    strategy_id: str
