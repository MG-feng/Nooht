from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ResourceRequest:
    vram_mb: int = 0
    cpu_cores: int = 1
    ram_mb: int = 512
    device_type: str = "auto"

class NoohtPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def version(self) -> str: ...
    @abstractmethod
    def get_resource_request(self) -> ResourceRequest: ...
    @abstractmethod
    async def init(self, config: Dict[str, Any]) -> bool: ...
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...
    @abstractmethod
    async def shutdown(self) -> bool: ...
