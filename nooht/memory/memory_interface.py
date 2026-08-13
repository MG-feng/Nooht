from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import time, uuid
class MemoryLevel(Enum): L0_RAM="l0_ram"; L1_SHARED="l1_shared"; L2_DISK="l2_disk"
@dataclass
class MemoryEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())); level: MemoryLevel = MemoryLevel.L0_RAM
    data: Any = None; metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time); last_accessed: float = field(default_factory=time.time)
    def estimate_bytes(self) -> int:
        from .memory_tracker import MemoryFootprintTracker
        return MemoryFootprintTracker.estimate_bytes(self.data)
@dataclass
class MemoryQuery: query_vector: Any = None; top_k: int = 5
