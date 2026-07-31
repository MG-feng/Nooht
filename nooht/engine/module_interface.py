from abc import ABC, abstractmethod
from typing import Dict, Any
from enum import Enum
import uuid

class ModuleStatus(Enum):
    UNINITIALIZED = "uninitialized"; READY = "ready"; RUNNING = "running"; ERROR = "error"; STOPPED = "stopped"

class NoohtModule(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name; self.module_id = str(uuid.uuid4())[:8]; self.config = config; self.status = ModuleStatus.UNINITIALIZED
    @abstractmethod
    async def initialize(self) -> bool: pass
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]: pass
    @abstractmethod
    async def cleanup(self) -> bool: pass
