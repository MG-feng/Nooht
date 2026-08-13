from typing import Dict, Any
from pydantic import BaseModel, Field

class SchedulerConfig(BaseModel): levels: int = Field(default=6, ge=1, le=10)
class DeviceConfig(BaseModel): backend: str = Field(default="auto")
class NRTConfig(BaseModel):
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
class NRTSchema(BaseModel):
    version: str = Field(default="1.0.0")
    nrt: NRTConfig = Field(default_factory=NRTConfig)
    workers: int = Field(default=1, ge=1)
def load_config(data: Dict[str, Any]) -> NRTSchema: return NRTSchema(**data)
